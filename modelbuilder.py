import torch
from torch import nn
from config import get_model_config
from loss.weightedmseloss import WeightedMSELoss
from loss.weightedmultioutputsloss import WeightedMultiOutputLoss
from loss.weightedrmseloss import WeightedRMSELoss
from model.Hernandez2021cnnlstm import Hernandez2021CNNLSTM
from model.bilstmmodel import BiLSTMModel
from model.transformer_tsai import TransformerTSAI
from utils.update_config import update_model_config


class ModelBuilder:
    def __init__(self, config):
        self.config = config
        self.n_input_channel = len(self.config['selected_sensors'])*6
        self.n_output = len(self.config['selected_opensim_labels'])
        self.model_name = self.config['model_name']
        self.model_config = get_model_config(f'config_{self.model_name}')
        self.model_config = update_model_config(self.config, self.model_config)
        self.optimizer_name = self.config['optimizer_name']
        self.learning_rate = self.config['learning_rate']
        self.l2_weight_decay_status = self.config['l2_weight_decay_status']
        self.l2_weight_decay = self.config['l2_weight_decay']
        self.loss = self.config['loss']
        self.weight = self.config['loss_weight']
        self.device = self.config['device']
        # self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        if not self.n_output == len(self.weight):
            self.weight = None

    def run_model_builder(self):
        model = self.get_model_architecture()
        criterion = self.get_criterion(self.weight)
        optimizer = self.get_optimizer()
        return model, optimizer, criterion

    def get_model_architecture(self):
        if self.model_name == 'bilstm': # done
            self.model = BiLSTMModel(self.model_config)
        elif self.model_name == 'hernandez2021cnnlstm': # done
            self.model = Hernandez2021CNNLSTM(self.model_config)
        elif self.model_name == 'transformertsai':
            c_in = self.n_input_channel  # aka channels, features, variables, dimensions
            c_out = self.n_output
            seq_len = self.config['target_padding_length']
            y_range = self.config['target_padding_length']
            max_seq_len = self.config['target_padding_length']
            d_model = self.model_config['tsai_d_model']
            n_heads = self.model_config['tsai_n_heads']
            d_k = d_v = None  # if None --> d_model // n_heads
            d_ff = self.model_config['tsai_d_ff']
            res_dropout = self.model_config['tsai_res_dropout_p']
            activation = "gelu"
            n_layers = self.model_config['tsai_n_layers']
            fc_dropout = self.model_config['tsai_fc_dropout_p']
            classification = self.model_config['classification']
            kwargs = {}
            self.model = TransformerTSAI(c_in, c_out, seq_len, max_seq_len=max_seq_len, d_model=d_model, n_heads=n_heads,
                            d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout, act=activation, n_layers=n_layers,
                            fc_dropout=fc_dropout, classification=classification,  **kwargs)
        return self.model

    def get_optimizer(self):
        if self.optimizer_name == 'Adam':
            if self.l2_weight_decay_status:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight_decay)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self.optimizer

    def get_criterion(self, weight=None):
        if self.loss == 'RMSE' and weight is not None:
            weight = torch.tensor(weight).to(self.device)
            self.criterion = WeightedRMSELoss(weight)
        elif self.loss == 'RMSE' and weight is None:
            self.criterion = torch.sqrt(nn.MSELoss())
        elif self.loss == 'MSE' and weight is not None:
            weight = torch.tensor(weight).to(self.device)
            self.criterion = WeightedMSELoss(weight)
        elif self.loss == 'MSE-CE' and weight is not None:
            weight = torch.tensor(weight).to(self.device)
            self.criterion = WeightedMultiOutputLoss(weight)
        else:
            self.criterion = nn.MSELoss()
        return self.criterion


