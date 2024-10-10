import torch.nn as nn
import torch
import torch.nn.functional as F
from src.models.ctd_nko import CTD_NKO
from torch.nn.utils import clip_grad_norm_
import numpy as np
from src.utils.utils import advanced_indexing_shuffle_3d
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Function

def grad_reverse(x, scale=1.0):

    class ReverseGrad(Function):
        """
        Gradient reversal layer
        """

        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad_output):
            return scale * grad_output.neg()

    return ReverseGrad.apply(x)

class CTD_NKO_BR(CTD_NKO):
    def __init__(self, dataset_collection, config):
        super().__init__(dataset_collection, config)

        self.current_train_dataloader = DataLoader(self.dataset_collection.train_f, shuffle=True, batch_size=self.batch_size)
        self.current_val_dataloader = DataLoader(self.dataset_collection.val_f, shuffle=False, batch_size=self.val_batch_size)

    def split_parameters(self):
        # split the parameters into three parts
        other_parameters = []
        if self.transpose:
            other_parameters.extend(self.transpose_net.parameters())
        other_parameters.extend(self.coefs_encoder.parameters())
        other_parameters.extend(self.koopman_operator_encoder.parameters())
        other_parameters.extend(self.koopman_linear.parameters())
        other_parameters.extend(self.G_y.parameters())
        other_parameters.extend(self.G_w.parameters())
        other_parameters.extend(self.ema_net_y.parameters())
        other_parameters.extend(self.G_x.parameters())
        other_parameters.extend(self.ema_net_x.parameters())
        other_parameters.append(self.global_koop)
        if self.lambda_DD > 0:
            balancing_params = list(self.D_net.parameters())
        else:
            balancing_params = []
        return other_parameters, balancing_params

    def configure_optimizers(self):
        if self.lambda_DD > 0:
            other_parameters, balancing_params = self.split_parameters()
            optimizer_D = torch.optim.Adam(balancing_params, lr=self.lr_D, weight_decay=self.weight_decay_D)
            optimizer_O = torch.optim.Adam(other_parameters, lr=self.lr, weight_decay=self.weight_decay)

            return [optimizer_D, optimizer_O]
        else:
            # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            other_parameters, balancing_params = self.split_parameters()
            optimizer = torch.optim.Adam(other_parameters, lr=self.lr, weight_decay=self.weight_decay)
            return optimizer

    def init_params(self):
        self.init_exp_params()
        self.init_dataset_params()
        self.init_model_params()
        self.lambda_DD = self.config['exp']['lambda_DD']

    def init_model(self):
        self.init_model_()
        if self.lambda_DD > 0:
            self.hiddens_D_net = self.config['model']['hiddens_D_net']
            self.lr_D = self.config['exp']['lr_D']
            self.weight_decay_D = self.config['exp']['weight_decay_D']
        
            self.D_net = nn.Sequential()
            input_size = self.d_model
            output_size = self.treatment_size
            for i in range(len(self.hiddens_D_net)):
                if i == 0:
                    self.D_net.add_module('fc{}'.format(i), nn.Linear(input_size, self.hiddens_D_net[i]))
                else:
                    if self.config['model']['activation'] == 'relu':
                        self.D_net.add_module('relu{}'.format(i), nn.ReLU())
                    else:
                        self.D_net.add_module('elu{}'.format(i), nn.ELU())
                    self.D_net.add_module('fc{}'.format(i), nn.Linear(self.hiddens_D_net[i-1], self.hiddens_D_net[i]))
            if self.config['model']['activation'] == 'relu':
                self.D_net.add_module('relu{}'.format(len(self.hiddens_D_net)), nn.ReLU())
            else:
                self.D_net.add_module('elu{}'.format(len(self.hiddens_D_net)), nn.ELU())
            self.D_net.add_module('fc{}'.format(len(self.hiddens_D_net)), nn.Linear(self.hiddens_D_net[-1], output_size))
            self.balancing = self.config['exp']['balancing']

    
    def get_a_hat(self, batch, update_D=True):
        # get the predicted A
        # if update_D is True, we will only update the D_net
        br = self.build_hidden_state(batch)
        if update_D:
            # we will not update the hidden_net if update_D
            br = br.detach()
        if self.balancing == 'grad_reverse':
            br = grad_reverse(br, self.lambda_D * self.lambda_DD)

        a_hat = self.D_net(br)
        return a_hat

    def bce(self, treatment_pred, current_treatments, active_entries):
        if self.treatment_mode == 'multiclass':
            loss = F.cross_entropy(treatment_pred.permute(0, 2, 1), current_treatments.permute(0, 2, 1), reduce=False)
            loss = loss.unsqueeze(-1)
        elif self.treatment_mode == 'multilabel':
            loss = F.binary_cross_entropy_with_logits(treatment_pred, current_treatments, reduce=False)
        else:
            raise NotImplementedError()
        loss = torch.sum(loss * active_entries) / torch.sum(active_entries)
        return loss

    def bce_loss(self, treatment_pred, current_treatments, active_entries, kind='predict'):
        if kind == 'predict':
            bce_loss = self.bce(treatment_pred, current_treatments, active_entries)
        elif kind == 'confuse':
            uniform_treatments = torch.ones_like(current_treatments)
            if self.treatment_mode == 'multiclass':
                uniform_treatments *= 1 / current_treatments.shape[-1]
            elif self.treatment_mode == 'multilabel':
                uniform_treatments *= 0.5
            bce_loss = self.bce(treatment_pred, uniform_treatments, active_entries)
        else:
            raise NotImplementedError()
        return bce_loss

    def training_step(self, batch, batch_idx):
        if not self.init_ema_tag:
            self.init_ema()
            self.init_ema_tag = True

        if self.lambda_DD > 0:
            optimizer_D, optimizer_O = self.optimizers()
        else:
            optimizer_O = self.optimizers()
        active_entries = batch['active_entries'] if 'active_entries' in batch else torch.ones_like(output)

        # update the other parameters
        self.toggle_optimizer(optimizer_O)
        
        y_x_hat, loss_koopman = self(batch)

        y_hat, x_hat = y_x_hat[:, :, :self.output_size], y_x_hat[:, :, self.output_size:]
        output = batch['outputs']
        loss_y = self.get_mse_all(y_hat, output, active_entries)

        if self.predict_X:
            next_covariates = batch['next_vitals']
            x_hat = x_hat[:, :next_covariates.shape[1], :]
            if self.loss_type_X == 'l1':
                loss_x = self.get_l1_all(x_hat, next_covariates, active_entries[:, 1:])
            elif self.loss_type_X == 'l2':
                loss_x = self.get_mse_all(x_hat, next_covariates, active_entries[:, 1:]) 
            else:
                raise ValueError('loss_type_X should be one of l1 and l2')
        else:
            loss_x = 0

        if self.lambda_D * self.lambda_DD > 0:
            a_hat = self.get_a_hat(batch, False)
            current_treatments = batch['current_treatments']
            if self.balancing == 'grad_reverse':
                loss_D = self.bce_loss(a_hat, current_treatments, active_entries)
            else:
                loss_D = self.bce_loss(a_hat, current_treatments, active_entries, kind='confuse') * self.lambda_D * self.lambda_DD
        else:
            loss_D = 0

        loss = self.lambda_Y * loss_y + self.lambda_X * loss_x + self.lambda_D * loss_koopman + loss_D 

        self.manual_backward(loss)
        # clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer_O.step()
        optimizer_O.zero_grad()
        self.untoggle_optimizer(optimizer_O)

        if self.weights_ema:
            self.ema.update()
            
        self.log('train_loss', loss, on_epoch=True)

        if self.lambda_D * self.lambda_DD > 0:
            # update the discriminator D
            self.toggle_optimizer(optimizer_D)
            a_hat = self.get_a_hat(batch)
            current_treatments = batch['current_treatments']
            if self.balancing == 'grad_reverse':
                loss_D = self.bce_loss(a_hat, current_treatments, active_entries)
            else:
                loss_D = self.bce_loss(a_hat, current_treatments, active_entries)

            self.manual_backward(loss_D)
            clip_grad_norm_(self.D_net.parameters(), max_norm=1.0)
            optimizer_D.step()
            optimizer_D.zero_grad()
            # if self.weights_ema:
            #     self.ema.update()
            self.untoggle_optimizer(optimizer_D)
        
        return {'loss': loss, 'loss_x': loss_x}

    def validation_step(self, batch, batch_idx):
        if self.weights_ema:
            with self.ema.average_parameters():
                y_x_hat, loss_koopman = self.forward(batch)
        else:
            y_x_hat, loss_koopman = self.forward(batch)

        active_entries = batch['active_entries'] if 'active_entries' in batch else torch.ones_like(output)

        y_hat, x_hat = y_x_hat[:, :, :self.output_size], y_x_hat[:, :, self.output_size:]
        output = batch['outputs']
        loss_y = self.get_mse_all(y_hat, output, active_entries)
        # loss_y = self.get_mse_all(y_hat, output, active_entries)
        if self.predict_X:
            next_covariates = batch['next_vitals']
            x_hat = x_hat[:, :next_covariates.shape[1], :]
            if self.loss_type_X == 'l1':
                loss_x = self.get_l1_all(x_hat, next_covariates, active_entries[:, 1:])
            elif self.loss_type_X == 'l2':
                loss_x = self.get_mse_all(x_hat, next_covariates, active_entries[:, 1:])
            else:
                raise ValueError('loss_type_X should be one of l1 and l2')
        else:
            loss_x = 0

        loss = self.lambda_Y * loss_y + self.lambda_X * loss_x + self.lambda_D * loss_koopman

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_loss_x', loss_x, on_epoch=True)
        self.log('val_loss_y', loss_y, on_epoch=True)
        self.log('val_loss_koopman', loss_koopman, on_epoch=True)