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
        # self.bnm1 = nn.BatchNorm1d(self.n_input_channel, momentum=0.1)
        self.lstm1 = nn.LSTM(input_size=self.n_input_channel, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.input_size1 = self.hidden_size*self.sequence_length*2
        # self.bnm2 = nn.BatchNorm1d(self.hidden_size*2, momentum=0.1)
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
        elif self.n_output == 4:
            self.fc2_1 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_2 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_3 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_4 = nn.Linear(self.output_size1, self.output_size2)
        elif self.n_output == 5:
            self.fc2_1 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_2 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_3 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_4 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_5 = nn.Linear(self.output_size1, self.output_size2)
        elif self.n_output == 6:
            self.fc2_1 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_2 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_3 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_4 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_5 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_6 = nn.Linear(self.output_size1, self.output_size2)
        elif self.n_output == 7:
            self.fc2_1 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_2 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_3 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_4 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_5 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_6 = nn.Linear(self.output_size1, self.output_size2)
            self.fc2_7 = nn.Linear(self.output_size1, self.output_size2)
        elif self.n_output == 8:
            self.fc2_1 = nn.Linear(128, 128)
            self.fc2_2 = nn.Linear(128, 128)
            self.fc2_3 = nn.Linear(128, 128)
            self.fc2_4 = nn.Linear(128, 128)
            self.fc2_5 = nn.Linear(128, 128)
            self.fc2_6 = nn.Linear(128, 128)
            self.fc2_7 = nn.Linear(128, 128)
            self.fc2_8 = nn.Linear(128, 128)
        elif self.n_output == 9:
            self.fc2_1 = nn.Linear(128, 128)
            self.fc2_2 = nn.Linear(128, 128)
            self.fc2_3 = nn.Linear(128, 128)
            self.fc2_4 = nn.Linear(128, 128)
            self.fc2_5 = nn.Linear(128, 128)
            self.fc2_6 = nn.Linear(128, 128)
            self.fc2_7 = nn.Linear(128, 128)
            self.fc2_8 = nn.Linear(128, 128)
            self.fc2_9 = nn.Linear(128, 128)
        elif self.n_output == 10:
            self.fc2_1 = nn.Linear(128, 128)
            self.fc2_2 = nn.Linear(128, 128)
            self.fc2_3 = nn.Linear(128, 128)
            self.fc2_4 = nn.Linear(128, 128)
            self.fc2_5 = nn.Linear(128, 128)
            self.fc2_6 = nn.Linear(128, 128)
            self.fc2_7 = nn.Linear(128, 128)
            self.fc2_8 = nn.Linear(128, 128)
            self.fc2_9 = nn.Linear(128, 128)
            self.fc2_10 = nn.Linear(128, 128)
        elif self.n_output == 11:
            self.fc2_1 = nn.Linear(128, 128)
            self.fc2_2 = nn.Linear(128, 128)
            self.fc2_3 = nn.Linear(128, 128)
            self.fc2_4 = nn.Linear(128, 128)
            self.fc2_5 = nn.Linear(128, 128)
            self.fc2_6 = nn.Linear(128, 128)
            self.fc2_7 = nn.Linear(128, 128)
            self.fc2_8 = nn.Linear(128, 128)
            self.fc2_9 = nn.Linear(128, 128)
            self.fc2_10 = nn.Linear(128, 128)
            self.fc2_11 = nn.Linear(128, 128)
        elif self.n_output == 12:
            self.fc2_1 = nn.Linear(128, 128)
            self.fc2_2 = nn.Linear(128, 128)
            self.fc2_3 = nn.Linear(128, 128)
            self.fc2_4 = nn.Linear(128, 128)
            self.fc2_5 = nn.Linear(128, 128)
            self.fc2_6 = nn.Linear(128, 128)
            self.fc2_7 = nn.Linear(128, 128)
            self.fc2_8 = nn.Linear(128, 128)
            self.fc2_9 = nn.Linear(128, 128)
            self.fc2_10 = nn.Linear(128, 128)
            self.fc2_11 = nn.Linear(128, 128)
            self.fc2_12 = nn.Linear(128, 128)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # x = self.bnm1(x)
        # x = x.permute(0, 2, 1)
        out, (h, c) = self.lstm1(x)
        # out = out.permute(0, 2, 1)
        # out = self.bnm2(out)
        # out = out.permute(0, 2, 1)
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
        elif self.n_output == 4:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out2 = F.relu(out)
            out2 = self.fc2_2(out2)
            out3 = F.relu(out)
            out3 = self.fc2_3(out3)
            out4 = F.relu(out)
            out4 = self.fc2_4(out4)
            out = [out1, out2, out3, out4]
        elif self.n_output == 5:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out2 = F.relu(out)
            out2 = self.fc2_2(out2)
            out3 = F.relu(out)
            out3 = self.fc2_3(out3)
            out4 = F.relu(out)
            out4 = self.fc2_4(out4)
            out5 = F.relu(out)
            out5 = self.fc2_5(out5)
            out = [out1, out2, out3, out4, out5]
        elif self.n_output == 6:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out2 = F.relu(out)
            out2 = self.fc2_2(out2)
            out3 = F.relu(out)
            out3 = self.fc2_3(out3)
            out4 = F.relu(out)
            out4 = self.fc2_4(out4)
            out5 = F.relu(out)
            out5 = self.fc2_4(out5)
            out6 = F.relu(out)
            out6 = self.fc2_6(out6)
            out = [out1, out2, out3, out4, out5, out6]
        elif self.n_output == 7:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out2 = F.relu(out)
            out2 = self.fc2_2(out2)
            out3 = F.relu(out)
            out3 = self.fc2_3(out3)
            out4 = F.relu(out)
            out4 = self.fc2_4(out4)
            out5 = F.relu(out)
            out5 = self.fc2_4(out5)
            out6 = F.relu(out)
            out6 = self.fc2_6(out6)
            out7 = F.relu(out)
            out7 = self.fc2_7(out7)
            out = [out1, out2, out3, out4, out5, out6, out7]
        elif self.n_output == 8:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out2 = F.relu(out)
            out2 = self.fc2_2(out2)
            out3 = F.relu(out)
            out3 = self.fc2_3(out3)
            out4 = F.relu(out)
            out4 = self.fc2_4(out4)
            out5 = F.relu(out)
            out5 = self.fc2_4(out5)
            out6 = F.relu(out)
            out6 = self.fc2_6(out6)
            out7 = F.relu(out)
            out7 = self.fc2_7(out7)
            out8 = F.relu(out)
            out8 = self.fc2_8(out8)
            out = [out1, out2, out3, out4, out5, out6, out7, out8]
        elif self.n_output == 9:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out2 = F.relu(out)
            out2 = self.fc2_2(out2)
            out3 = F.relu(out)
            out3 = self.fc2_3(out3)
            out4 = F.relu(out)
            out4 = self.fc2_4(out4)
            out5 = F.relu(out)
            out5 = self.fc2_4(out5)
            out6 = F.relu(out)
            out6 = self.fc2_6(out6)
            out7 = F.relu(out)
            out7 = self.fc2_7(out7)
            out8 = F.relu(out)
            out8 = self.fc2_8(out8)
            out9 = F.relu(out)
            out9 = self.fc2_9(out9)
            out = [out1, out2, out3, out4, out5, out6, out7, out8, out9]
        elif self.n_output == 10:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out2 = F.relu(out)
            out2 = self.fc2_2(out2)
            out3 = F.relu(out)
            out3 = self.fc2_3(out3)
            out4 = F.relu(out)
            out4 = self.fc2_4(out4)
            out5 = F.relu(out)
            out5 = self.fc2_4(out5)
            out6 = F.relu(out)
            out6 = self.fc2_6(out6)
            out7 = F.relu(out)
            out7 = self.fc2_7(out7)
            out8 = F.relu(out)
            out8 = self.fc2_8(out8)
            out9 = F.relu(out)
            out9 = self.fc2_9(out9)
            out10 = F.relu(out)
            out10 = self.fc2_10(out10)
            out = [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10]
        elif self.n_output == 11:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out2 = F.relu(out)
            out2 = self.fc2_2(out2)
            out3 = F.relu(out)
            out3 = self.fc2_3(out3)
            out4 = F.relu(out)
            out4 = self.fc2_4(out4)
            out5 = F.relu(out)
            out5 = self.fc2_4(out5)
            out6 = F.relu(out)
            out6 = self.fc2_6(out6)
            out7 = F.relu(out)
            out7 = self.fc2_7(out7)
            out8 = F.relu(out)
            out8 = self.fc2_8(out8)
            out9 = F.relu(out)
            out9 = self.fc2_9(out9)
            out10 = F.relu(out)
            out10 = self.fc2_10(out10)
            out11 = F.relu(out)
            out11 = self.fc2_11(out11)
            out = [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11]
        elif self.n_output == 12:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out2 = F.relu(out)
            out2 = self.fc2_2(out2)
            out3 = F.relu(out)
            out3 = self.fc2_3(out3)
            out4 = F.relu(out)
            out4 = self.fc2_4(out4)
            out5 = F.relu(out)
            out5 = self.fc2_4(out5)
            out6 = F.relu(out)
            out6 = self.fc2_6(out6)
            out7 = F.relu(out)
            out7 = self.fc2_7(out7)
            out8 = F.relu(out)
            out8 = self.fc2_8(out8)
            out9 = F.relu(out)
            out9 = self.fc2_9(out9)
            out10 = F.relu(out)
            out10 = self.fc2_10(out10)
            out11 = F.relu(out)
            out11 = self.fc2_11(out11)
            out12 = F.relu(out)
            out12 = self.fc2_12(out12)
            out = [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12]
        return torch.stack(out, dim=0).permute(1, 2, 0)