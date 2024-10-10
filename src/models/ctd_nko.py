import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch_ema import ExponentialMovingAverage
import logging
from typing import List
from thop import clever_format, profile
from fvcore.nn import FlopCountAnalysis
from src.utils.utils import *
from src.models.wass_calculator import get_weights
import psutil
process = psutil.Process()
import os
from hydra.utils import get_original_cwd

class CTD_NKO(pl.LightningModule):
    def __init__(self, dataset_collection, config):
        super().__init__()
        self.dataset_collection = dataset_collection
        self.config = config
        self.init_params()
        self.init_model()
        self.init_ema()
        self.count_flops_processed = False
        self.automatic_optimization = False
        self.init_ema_tag = False
        self.save_hyperparameters('config')
        self.initialize_weights()
        self.logged_memory = False
        if config['exp']['mode'] != 'tune':
            self.txt_dir = os.path.join(get_original_cwd(), config.exp.csv_dir).replace('csvs', 'memory').replace('.0', '')
            self.txt_path = os.path.join(self.txt_dir, f'{config.model.name}.txt')

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def init_ema(self):
        if self.weights_ema:
            parameters = [par for par in self.parameters()]
            self.ema = ExponentialMovingAverage([par for par in parameters], decay=self.beta)
    
    def init_params(self):
        self.init_exp_params()
        self.init_dataset_params()
        self.init_model_params()

    def init_dataset_params(self):
        self.max_seq_length = self.config['dataset']['max_seq_length']
        self.treatment_size = self.config['dataset']['treatment_size']
        self.one_hot_treatment_size = self.config['dataset']['one_hot_treatment_size']
        self.static_size = self.config['dataset']['static_size']
        self.output_size = self.config['dataset']['output_size']
        self.input_size = self.config['dataset']['input_size']
        self.treatment_mode = self.config['dataset']['treatment_mode']
        self.autoregressive = self.config['dataset']['autoregressive']
        self.val_batch_size = self.config['dataset']['val_batch_size']
        self.projection_horizon = self.config['dataset']['projection_horizon']
        self.predict_X = self.config['dataset']['predict_X']

    def init_exp_params(self):
        self.lr = self.config['exp']['lr']
        self.lr_D = self.config['exp']['lr_D']
        self.weight_decay = self.config['exp']['weight_decay']
        self.weight_decay_D = self.config['exp']['weight_decay_D']
        self.patience = self.config['exp']['sch_patience']
        self.patience_D = self.config['exp']['sch_patience_D']

        self.factor = self.config['exp']['factor']
        self.batch_size = self.config['exp']['batch_size']
        self.dropout = self.config['exp']['dropout']
        self.cooldown = self.config['exp']['cooldown']
        self.weights_ema = self.config['exp']['weights_ema']
        self.beta = self.config['exp']['beta']
        self.update_lambda_D = self.config['exp']['update_lambda_D']
        self.lambda_D = self.config['exp']['lambda_D'] if not self.update_lambda_D else 0.0
        self.lambda_D_max = self.config['exp']['lambda_D']
        # self.lambda_W = self.config['exp']['lambda_W']
        self.lambda_X = self.config['exp']['lambda_X']
        self.lambda_Y = self.config['exp']['lambda_Y']
        self.loss_type_X = self.config['exp']['loss_type_X']
        self.epochs = self.config['exp']['epochs']
        self.ema_y = self.config['exp']['ema_y']
        self.ema_hidden = self.config['exp']['ema_hidden']

    def init_model_params(self):
        self.use_global = self.config['model']['use_global']
        self.transpose = self.config['model']['transpose']
        if self.transpose:
            self.transpose_size = self.config['model']['transpose_size']
        self.num_sins = self.config['model']['num_sins']
        self.num_poly = self.config['model']['num_poly']
        self.num_exp = self.config['model']['num_exp']
        self.d_model = self.config['model']['d_model']
        self.num_layers = self.config['model']['num_layers']
        self.koopman_dim = self.config['model']['koopman_dim_add'] + self.d_model
        self.num_layers_koopman = self.config['model']['num_layers_koopman']

        self.hiddens_G_w = self.config['model']['hiddens_G_w']

        self.dims_hidden = self.config['model']['dims_hidden']
        self.hiddens_G_w = self.config['model']['hiddens_G_w']
        self.hiddens_G_y = self.config['model']['hiddens_G_y']
        self.hiddens_G_x = self.config['model']['hiddens_G_x']

    def init_model(self):
        self.init_model_()

    def init_model_(self):
        if self.static_size > 0:
            input_size = self.input_size + self.static_size + self.treatment_size
        else:
            input_size = self.input_size + self.treatment_size

        if self.autoregressive:
            input_size += self.output_size

        if self.transpose:
            self.transpose_net = nn.Linear(input_size, self.transpose_size)
            input_size = self.transpose_size
        
        self.coefs_encoder = nn.LSTM(input_size, self.d_model + self.num_sins * 2, self.num_layers, dropout=self.dropout, batch_first=True)
        self.koopman_operator_encoder = nn.LSTM(input_size, self.koopman_dim, self.num_layers_koopman, dropout=self.dropout, batch_first=True)

        if -1 not in self.dims_hidden:
            self.koopman_linear = nn.Sequential()
            for i in range(len(self.dims_hidden)):
                if i == 0:
                    self.koopman_linear.add_module('fc{}'.format(i), nn.Linear(self.koopman_dim, self.dims_hidden[i]))
                else:
                    self.koopman_linear.add_module('elu{}'.format(i), nn.ELU())
                    self.koopman_linear.add_module('fc{}'.format(i), nn.Linear(self.dims_hidden[i-1], self.dims_hidden[i]))
            self.koopman_linear.add_module('elu{}'.format(len(self.dims_hidden)), nn.ELU())
            self.koopman_linear.add_module('fc{}'.format(len(self.dims_hidden)), nn.Linear(self.dims_hidden[-1], self.d_model * self.d_model))
        else:
            self.koopman_linear = nn.Linear(self.koopman_dim, self.d_model * self.d_model)

        # init the G_w net to predict W 
        self.G_w = nn.Sequential()
        if -1 not in self.hiddens_G_w:
            for i in range(len(self.hiddens_G_w)):
                if i == 0:
                    self.G_w.add_module('fc{}'.format(i), nn.Linear(self.d_model, self.hiddens_G_w[i]))
                else:
                    self.G_w.add_module('elu{}'.format(i), nn.ELU())
                    self.G_w.add_module('fc{}'.format(i), nn.Linear(self.hiddens_G_w[i-1], self.hiddens_G_w[i]))
            self.G_w.add_module('elu{}'.format(len(self.hiddens_G_w)), nn.ELU())
            self.G_w.add_module('fc{}'.format(len(self.hiddens_G_w)), nn.Linear(self.hiddens_G_w[-1], 1))
        else:
            self.G_w.add_module('fc{}'.format(1), nn.Linear(self.d_model, 1))
        # add the sigmoid layer to the end of the G_w net
        self.G_w.add_module('sigmoid{}'.format(1), nn.Sigmoid())

        # init the G_y net to predict Y
        self.G_y = nn.Sequential()
        if -1 not in self.hiddens_G_y:
            for i in range(len(self.hiddens_G_y)):
                if i == 0:
                    self.G_y.add_module('fc{}'.format(i), nn.Linear(self.d_model, self.hiddens_G_y[i]))
                else:
                    self.G_y.add_module('elu{}'.format(i), nn.ELU())
                    self.G_y.add_module('fc{}'.format(i), nn.Linear(self.hiddens_G_y[i-1], self.hiddens_G_y[i]))
            self.G_y.add_module('elu{}'.format(len(self.hiddens_G_y)), nn.ELU())
            self.G_y.add_module('fc{}'.format(len(self.hiddens_G_y)), nn.Linear(self.hiddens_G_y[-1], self.output_size))
        else:
            self.G_y.add_module('fc{}'.format(1), nn.Linear(self.d_model, self.output_size))

        # init the G_x net to predict X if needed
        if self.predict_X:
            self.G_x = nn.Sequential()
            if -1 not in self.hiddens_G_x:
                for i in range(len(self.hiddens_G_x)):
                    if i == 0:
                        self.G_x.add_module('fc{}'.format(i), nn.Linear(self.d_model, self.hiddens_G_x[i]))
                    else:
                        self.G_x.add_module('elu{}'.format(i), nn.ELU())
                        self.G_x.add_module('fc{}'.format(i), nn.Linear(self.hiddens_G_x[i-1], self.hiddens_G_x[i]))
                self.G_x.add_module('elu{}'.format(len(self.hiddens_G_x)), nn.ELU())
                self.G_x.add_module('fc{}'.format(len(self.hiddens_G_x)), nn.Linear(self.hiddens_G_x[-1], self.input_size))
            else:
                self.G_x.add_module('fc{}'.format(1), nn.Linear(self.d_model, self.input_size))
        else:
            self.G_x = nn.Identity()

        if self.predict_X:
            self.ema_net_x = nn.Sequential()
            self.ema_net_x.add_module('fc{}'.format(1), nn.Linear(self.d_model, self.input_size))
            self.ema_net_x.add_module('sigmoid{}'.format(1), nn.Sigmoid())
        else:
            self.ema_net_x = nn.Identity()
        
        if self.ema_y:
            self.ema_net_y = nn.Sequential()
            self.ema_net_y.add_module('fc{}'.format(1), nn.Linear(self.d_model, self.output_size))
            self.ema_net_y.add_module('sigmoid{}'.format(1), nn.Sigmoid())
        else:
            self.ema_net_y = nn.Identity()

        self.global_koop = nn.Parameter(torch.randn(self.d_model, self.d_model))

    def train_dataloader(self) -> DataLoader:
        # return DataLoader(self.dataset_collection.train_f, shuffle=True, batch_size=self.batch_size)
        return self.current_train_dataloader

    def val_dataloader(self) -> DataLoader:
        # return DataLoader(self.dataset_collection.val_f, batch_size=self.val_batch_size)
        return self.current_val_dataloader

    def build_input(self, batch, pre_treat=True):
        if self.static_size > 0:
            if self.predict_X:
                x = batch['vitals']
                x = torch.cat((x, batch['static_features']), dim=-1)
            # when we don't predict x, we use static features as the current_covariates
            else:
                x = batch['static_features']
        # if we use autoregressive, we need to use the previous output as the input
        if self.autoregressive:
            prev_outputs = batch['prev_outputs']
            x = torch.cat((x, prev_outputs), dim=-1)
        if pre_treat:
            previous_treatments = batch['prev_treatments']
            x = torch.cat((x, previous_treatments), dim=-1)
        else:
            # if we don't use previous treatments, we use the current treatments as the input to build koopman operator
            current_treatments = batch['current_treatments']
            x = torch.cat((x, current_treatments), dim=-1)
        # transpose the input if needed
        if self.transpose:
            x = self.transpose_net(x)
        return x
    
    def build_coefs(self, batch):
        x = self.build_input(batch) # [n, T, d]
        coefs, _ = self.coefs_encoder(x) # [n, T, d + 2 * num_sins]
        return coefs
    
    def build_hidden_state(self, batch):
        coefs = self.build_coefs(batch) # [n, T, d_model + num_sins * 2]
        hidden_states = torch.zeros(coefs.shape[0], coefs.shape[1], self.d_model).to(coefs.device) # [n, T, d_model]
        # polynomial function
        for i in range(self.num_poly):
            hidden_states[:, :, i] = coefs[:, :, i]**(i + 1)
        # exponential function
        for i in range(self.num_poly, self.num_poly + self.num_exp): 
            hidden_states[:, :, i] = torch.exp(coefs[:, :, i])
        # sine/cos functions
        for i in range(self.num_poly + self.num_exp, self.num_poly + self.num_exp + self.num_sins):
            hidden_states[:, :, i] = coefs[:, :, self.num_sins * 2 + i] * torch.cos(coefs[:, :, i])
            hidden_states[:, :, i + self.num_sins] = coefs[:, :, self.num_sins * 3 + i] * torch.sin(coefs[:, :, self.num_sins + i])
        # the remaining ouputs are purely data-driven measurement functions.
        hidden_states[:, :, self.num_poly + self.num_exp + self.num_sins * 2:] = coefs[:, :, self.num_poly + self.num_exp + self.num_sins * 4:]
        return hidden_states
    
    def build_causal_koopman_operator(self, batch):
        x = self.build_input(batch, pre_treat=False)
        koopman_hidden_states, _ = self.koopman_operator_encoder(x)
        causal_koopman_operator = self.koopman_linear(koopman_hidden_states) # [n, T, d_model * d_model]
        # reshape the causal koopman operator
        causal_koopman_operator = causal_koopman_operator.reshape(-1, causal_koopman_operator.shape[1], self.d_model, self.d_model)
        if self.use_global:
            causal_koopman_operator += self.global_koop
        return causal_koopman_operator
    
    def forward(self, batch):
        hidden_states = self.build_hidden_state(batch) # [n, T, d_model]
        causal_koopman_operator = self.build_causal_koopman_operator(batch) # [n, T, d_model, d_model]
        # apply the causal koopman operator to the hidden states
        next_hidden_states = torch.einsum('ntij, ntj -> nti', causal_koopman_operator, hidden_states) # [n, T, d_model]
        # get the predicted Y
        y_hat = self.G_y(next_hidden_states) # [n, T, output_size]
        # get the predicted X if needed
        n, T, _ = next_hidden_states.shape
        x_hat = torch.zeros(n, T, self.input_size).to(self.device)
        if self.predict_X:
            x_hat = self.G_x(next_hidden_states)
            ema_xw = self.ema_net_x(next_hidden_states)
            x_hat = ema_xw * batch['vitals'] + (1 - ema_xw) * x_hat
        if self.ema_y:
            ema_yw = self.ema_net_y(next_hidden_states)
            y_hat = ema_yw * batch['prev_outputs'] + (1 - ema_yw) * y_hat

        true_next_hidden_states = hidden_states[:, 1:, :]
        exstimated_next_hidden_states = next_hidden_states[:, :-1, :]
        loss_koopman = F.mse_loss(exstimated_next_hidden_states, true_next_hidden_states)

        return torch.cat((y_hat, x_hat), dim=-1), loss_koopman

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def get_mse_at_follow_up_time(self, prediction, output, active_entries=None):
        # cauculate mse at follow up time
        mses = torch.sum(torch.sum((prediction - output) ** 2 * active_entries, dim=0), dim=-1) / torch.sum(torch.sum(active_entries, dim=0), dim=-1)
        return mses

    def get_mse_all(self, prediction, output, active_entries=None, w=None):
        if w is not None:
            # active_entries to float
            mses = torch.sum((prediction - output) ** 2 * active_entries * w) / torch.sum(active_entries * w)
        else:
            mses = torch.sum((prediction - output) ** 2 * active_entries) / torch.sum(active_entries)
        return mses

    def get_l1_all(self, prediction, output, active_entries=None):
        l1 = torch.sum(torch.abs(prediction - output) * active_entries) / torch.sum(active_entries)
        return l1

    def get_predictions(self, dataset: Dataset, logger=None) -> np.array:
        if logger is not None:
            logger.info(f'Predictions for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.hparams['config']['dataset']['val_batch_size'], shuffle=False)
        outcome_pred, next_covariates_pred = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return outcome_pred.numpy(), next_covariates_pred.numpy()

    def get_normalised_masked_rmse(self, dataset: Dataset, one_step_counterfactual=False, logger=None):
        # adapted from https://github.com/Valentyn1997/CausalTransformer
        if logger is not None:
            logger.info(f'RMSE calculation for {dataset.subset_name}.')
        outputs_scaled, _ = self.get_predictions(dataset, logger=logger)
        
        unscale = self.hparams['config']['exp']['unscale_rmse']
        percentage = self.hparams['config']['exp']['percentage_rmse']

        if unscale:
            output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
            outputs_unscaled = outputs_scaled * output_stds + output_means

            # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
            mse = ((outputs_unscaled - dataset.data['unscaled_outputs']) ** 2) * dataset.data['active_entries']
        else:
            # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
            mse = ((outputs_scaled - dataset.data['outputs']) ** 2) * dataset.data['active_entries']

        # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
        mse_orig = mse.sum(0).sum(-1) / dataset.data['active_entries'].sum(0).sum(-1)
        mse_orig = mse_orig.mean()
        rmse_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const

        # Masked averaging over all dimensions at once
        mse_all = mse.sum() / dataset.data['active_entries'].sum()
        rmse_normalised_all = np.sqrt(mse_all) / dataset.norm_const

        if percentage:
            rmse_normalised_orig *= 100.0
            rmse_normalised_all *= 100.0

        if one_step_counterfactual:
            # Only considering last active entry with actual counterfactuals
            num_samples, time_dim, output_dim = dataset.data['active_entries'].shape
            last_entries = dataset.data['active_entries'] - np.concatenate([dataset.data['active_entries'][:, 1:, :], np.zeros((num_samples, 1, output_dim))], axis=1)
            if unscale:
                mse_last = ((outputs_unscaled - dataset.data['unscaled_outputs']) ** 2) * last_entries
            else:
                mse_last = ((outputs_scaled - dataset.data['outputs']) ** 2) * last_entries

            mse_last = mse_last.sum() / last_entries.sum()
            rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const

            if percentage:
                rmse_normalised_last *= 100.0

            return rmse_normalised_orig, rmse_normalised_all, rmse_normalised_last

        return rmse_normalised_orig, rmse_normalised_all

    def get_normalised_n_step_rmses(self, dataset: Dataset, datasets_mc: List[Dataset] = None, logger=None):
        # adapted from https://github.com/Valentyn1997/CausalTransformer
        if logger is not None:
            logger.info(f'RMSE calculation for {dataset.subset_name}.')
        assert hasattr(dataset, 'data_processed_seq')

        unscale = self.hparams['config']['exp']['unscale_rmse']
        percentage = self.hparams['config']['exp']['percentage_rmse']
        outputs_scaled = self.get_autoregressive_predictions(dataset if datasets_mc is None else datasets_mc, logger=logger)

        if unscale:
            output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
            outputs_unscaled = outputs_scaled * output_stds + output_means

            mse = ((outputs_unscaled - dataset.data_processed_seq['unscaled_outputs']) ** 2) \
                * dataset.data_processed_seq['active_entries']
        else:
            mse = ((outputs_scaled - dataset.data_processed_seq['outputs']) ** 2) * dataset.data_processed_seq['active_entries']

        nan_idx = np.unique(np.where(np.isnan(dataset.data_processed_seq['outputs']))[0])
        not_nan = np.array([i for i in range(outputs_scaled.shape[0]) if i not in nan_idx])

        # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
        mse_orig = mse[not_nan].sum(0).sum(-1) / dataset.data_processed_seq['active_entries'][not_nan].sum(0).sum(-1)
        rmses_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const

        if percentage:
            rmses_normalised_orig *= 100.0

        return rmses_normalised_orig

    def get_autoregressive_predictions(self, dataset: Dataset, logger=None) -> np.array:
        # adapted from https://github.com/Valentyn1997/CausalTransformer
        if logger is not None:
            logger.info(f'Autoregressive Prediction for {dataset.subset_name}.')
        projection_horizon = self.hparams['config']['dataset']['projection_horizon']

        predicted_outputs = np.zeros((len(dataset), projection_horizon, self.output_size))

        for t in range(projection_horizon + 1):
            if logger is not None:
                logger.info(f't = {t + 1}')
            outputs_scaled, next_covariates_pred = self.get_predictions(dataset)

            for i in range(len(dataset)):
                split = int(dataset.data['future_past_split'][i])
                if t < projection_horizon:
                    if self.predict_X:
                        # replace the covariates in next step with the predicted covariates
                        dataset.data['vitals'][i, split + t, :] = next_covariates_pred[i, split - 1 + t, :]
                    dataset.data['prev_outputs'][i, split + t, :] = outputs_scaled[i, split - 1 + t, :]
                    pass

                if t > 0:
                    predicted_outputs[i, t - 1, :] = outputs_scaled[i, split - 1 + t, :]

        return predicted_outputs

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        if self.weights_ema:
            with self.ema.average_parameters():
                y_x_hat, _ = self.forward(batch)
        else:
            y_x_hat, _ = self.forward(batch)
        prediction = y_x_hat[:, :, :self.output_size]
        next_covariates = y_x_hat[:,:,self.output_size:]
        return prediction, next_covariates

    def on_train_epoch_end(self):
        if 'loss_x_epoch' in self.trainer.logged_metrics:
            print(f"Epoch {self.trainer.current_epoch}: Valid Loss_x = {self.trainer.logged_metrics['val_loss_x']:.4f}")
            print(f"Epoch {self.trainer.current_epoch}: Valid Loss_y = {self.trainer.logged_metrics['val_loss_y']:.4f}")
            print(f"Epoch {self.trainer.current_epoch}: Valid Loss_koopman = {self.trainer.logged_metrics['val_loss_koopman']:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        # save the ema state
        if self.weights_ema:
            checkpoint['ema_state'] = self.ema.state_dict()
            
    def on_load_checkpoint(self, checkpoint):
        # load the ema state
        if self.weights_ema:
            self.ema.load_state_dict(checkpoint['ema_state'])

    def generate_hidden_states(self, dataloader):
        self.eval()  # Set the model to evaluation mode
        hidden_states = []
        with torch.no_grad():  # Disable gradient computation
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                hidden_state = self.build_hidden_state(batch)
                hidden_states.append(hidden_state)
        # Concatenate all hidden states along the first dimension
        hidden_states = torch.cat(hidden_states, dim=0)
        self.train()
        return hidden_states

    def create_train_dataloader(self, update=True):
        if update:
            data_loader = DataLoader(self.dataset_collection.train_f, shuffle=False, batch_size=self.batch_size)
            hidden_states = self.generate_hidden_states(data_loader)
            current_treatments = self.dataset_collection.train_f.data['current_treatments']
            current_treatments = torch.tensor(current_treatments).to(self.device)
            shuffled_treatments = advanced_indexing_shuffle_3d(current_treatments)
            real_samples = torch.cat([hidden_states, current_treatments], dim=-1)
            fake_samples = torch.cat([hidden_states, shuffled_treatments], dim=-1)
            w = get_weights(real_samples, fake_samples, epoch=self.epoch_w).detach()
            self.dataset_collection.train_f.data['w'] = w
            
        return DataLoader(self.dataset_collection.train_f, shuffle=True, batch_size=self.batch_size)

    def create_val_dataloader(self, update=True):
        if update:
            data_loader = DataLoader(self.dataset_collection.val_f, shuffle=False, batch_size=self.val_batch_size)
            hidden_states = self.generate_hidden_states(data_loader)
            current_treatments = self.dataset_collection.val_f.data['current_treatments']
            current_treatments = torch.tensor(current_treatments).to(self.device)
            shuffled_treatments = advanced_indexing_shuffle_3d(current_treatments)
            real_samples = torch.cat([hidden_states, current_treatments], dim=-1)
            fake_samples = torch.cat([hidden_states, shuffled_treatments], dim=-1)
            w = get_weights(real_samples, fake_samples, epoch=self.epoch_w).detach()
            self.dataset_collection.val_f.data['w'] = w
            
        return DataLoader(self.dataset_collection.val_f, shuffle=False, batch_size=self.val_batch_size)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.logged_memory:
            peak_memory = torch.cuda.max_memory_allocated()
            print(f"Peak memory usage after first batch: {peak_memory / 1024**3} GB")
            self.logged_memory = True
            cpu_mem = process.memory_info().rss / 1024 / 1024 / 1024
            print(f"Current CPU memory usage = {cpu_mem:.2f} GB")
            if self.config['exp']['mode'] != 'tune':
                os.makedirs(os.path.dirname(self.txt_path), exist_ok=True)
                with open(self.txt_path, 'a') as f:
                    f.write(f"Peak memory usage after first batch: {peak_memory / 1024**3} GB\n")
                    f.write(f"Current CPU memory usage = {cpu_mem:.2f} GB\n")
        super().on_train_batch_end(outputs, batch, batch_idx)