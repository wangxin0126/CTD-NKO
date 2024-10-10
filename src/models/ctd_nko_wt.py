import torch.nn as nn
import torch
import torch.nn.functional as F
from src.models.ctd_nko import CTD_NKO
from torch.nn.utils import clip_grad_norm_
import numpy as np
from src.utils.utils import advanced_indexing_shuffle_3d
from torch.utils.data import DataLoader, TensorDataset


class CTD_NKO_WT(CTD_NKO):
    def __init__(self, dataset_collection, config):
        super().__init__(dataset_collection, config)

        self.current_train_dataloader = DataLoader(self.dataset_collection.train_f, shuffle=True, batch_size=self.batch_size)
        self.current_val_dataloader = DataLoader(self.dataset_collection.val_f, shuffle=False, batch_size=self.val_batch_size)

    def on_train_epoch_start(self):
        epoch = self.trainer.current_epoch

        # if epoch == 0:
        #     self.current_train_dataloader = DataLoader(self.dataset_collection.train_f, shuffle=True, batch_size=self.batch_size)
        #     self.current_val_dataloader = DataLoader(self.dataset_collection.val_f, shuffle=False, batch_size=self.val_batch_size)
        if epoch % self.update_freq == self.update_freq - 1:
            self.current_train_dataloader = self.create_train_dataloader()
            self.current_val_dataloader = self.create_val_dataloader()

    def init_params(self):
        self.init_exp_params()
        self.init_dataset_params()
        self.init_model_params()
        self.update_freq = self.config['exp']['update_freq']
        self.epoch_w = self.config['exp']['epoch_w']
            
    def training_step(self, batch, batch_idx):
        if not self.init_ema_tag:
            self.init_ema()
            self.init_ema_tag = True
        
        optimizer = self.optimizers()
        active_entries = batch['active_entries'] if 'active_entries' in batch else torch.ones_like(output)
        # Compute the MMD loss

        if 'w' in batch:
            w = batch['w']
            # print('w is provided')
            # print(w[0, 0])
        else:
            w = torch.ones_like(active_entries[:, :, 0:1]).to(active_entries.device)

        w = F.softmax(w, dim=0)

        y_x_hat, loss_koopman = self(batch)

        y_hat, x_hat = y_x_hat[:, :, :self.output_size], y_x_hat[:, :, self.output_size:]
        output = batch['outputs']
        loss_y = self.get_mse_all(y_hat, output, active_entries, w)

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

        self.manual_backward(loss)
        
        optimizer.step()
        optimizer.zero_grad()

        if self.weights_ema:
            self.ema.update()
        
        self.log('train_loss', loss, on_epoch=True)
        # self.log('loss_x', loss_x, on_epoch=True)
        # self.log('loss_y', loss_y, on_epoch=True)
        # self.log('loss_koopman', loss_koopman, on_epoch=True)
        return {'loss': loss, 'loss_x': loss_x}

    def validation_step(self, batch, batch_idx):
        if self.weights_ema:
            with self.ema.average_parameters():
                y_x_hat, loss_koopman = self.forward(batch)
        else:
            y_x_hat, loss_koopman = self.forward(batch)

        active_entries = batch['active_entries'] if 'active_entries' in batch else torch.ones_like(output)

        if 'w' in batch:
            w = batch['w']
        else:
            w = torch.ones_like(y_x_hat[:, :, 0:1]).to(y_x_hat.device)

        w = F.softmax(w, dim=0)

        y_hat, x_hat = y_x_hat[:, :, :self.output_size], y_x_hat[:, :, self.output_size:]
        output = batch['outputs']
        loss_y = self.get_mse_all(y_hat, output, active_entries, w)
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