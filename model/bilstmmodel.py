import torch
import torch.nn as nn
import torch.nn.functional as F
'''
References:
Sharifi Renani 2021:
hidden_size: 32 for hip and 128 for knee
num_layers:1
dropout_p: 0.5

Jay-Shian Tan 2022: Predicting Knee Joint Kinematics from Wearable Sensor Data in People with Knee Osteoarthritis and Clinical Considerations for Future Machine Learning Models
hidden_size: 128
num_layers:2
dropout_p: 0.2
'''

class BiLSTMModel(nn.Module):
    def __init__(self, config):
        super(BiLSTMModel, self).__init__()
        '''
        baseline hyper parameters:
        hidden_size: 32
        num_layers:3
        dropout_p: 0.5
        '''
        self.n_input_channel = len(config['selected_sensors'])*6
        self.n_output = len(config['selected_opensim_labels'])
        self.hidden_size = config['bilstm_hidden_size']
        self.num_layers = config['bilstm_num_layers']
        dropout_p = config['bilstm_dropout_p']
        self.dropout = nn.Dropout(p=dropout_p)

        self.sequence_length = config['target_padding_length']
        self.lstm1 = nn.LSTM(input_size=self.n_input_channel, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.input_size1 = self.hidden_size*self.sequence_length*2
        self.output_size1 = 1600
        self.fc1 = nn.Linear(self.input_size1, self.output_size1)  # 2 for bidirection
        self.output_size2 = self.sequence_length
        if self.n_output == 1:
            self.fc2_1 = nn.Linear(self.output_size1, self.output_size2)
        elif self.n_output == 2:
            self.fc2_1 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_2 = nn.Linear(self.output_size1, self.output_size2)
        elif self.n_output == 3:
            self.fc2_1 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_2 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_3 = nn.Linear(self.output_size1, self.output_size2)

    def forward(self, x):
        out, (h, c) = self.lstm1(x)
        out = torch.flatten(out, start_dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.dropout(out)
        if self.n_output == 1:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out = [out1]
        elif self.n_output == 2:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out2 = F.relu(out)
            out2 = self.fc2_2(out2)
            out = [out1, out2]
        elif self.n_output == 3:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out2 = F.relu(out)
            out2 = self.fc2_2(out2)
            out3 = F.relu(out)
            out3 = self.fc2_3(out3)
            out = [out1, out2, out3]
        return torch.stack(out, dim=0).permute(1, 2, 0)