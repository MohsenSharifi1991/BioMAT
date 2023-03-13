import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, config):
        super(EncoderCNN, self).__init__()
        # n_input_channel, target_length
        self.nn_input_length = config['target_padding_length']
        self.conv_l1_in_channel = config["EncoderCNN_conv_l1_in_channel"]
        self.conv_l1_in_width = config['EncoderCNN_conv_l1_in_width']
        self.conv_l1_out_channel = config["EncoderCNN_conv_l1_out_channel"]
        self.conv_l1_kernel_size = config["EncoderCNN_conv_l1_kernel_size"]
        self.conv_l1_dropout = config["EncoderCNN_conv_l1_dropout"]

        self.conv_l2_in_channel = self.conv_l1_out_channel
        self.conv_l2_out_channel = config["EncoderCNN_conv_l2_out_channel"]
        self.conv_l2_kernel_size = config["EncoderCNN_conv_l2_kernel_size"]
        self.conv_l2_dropout = config["EncoderCNN_conv_l2_dropout"]

        self.fc1_in_feature = config["EncoderCNN_cnn_fc1_in_feature"]
        self.fc1_out_feature = config["EncoderCNN_cnn_fc1_out_feature"]
        self.fc1_dropout = config["EncoderCNN_fc1_dropout"]

        self.conv1 = nn.Conv2d(self.conv_l1_in_channel, self.conv_l1_out_channel, kernel_size=self.conv_l1_kernel_size, padding=(1,1))
        self.conv2 = nn.Conv2d(self.conv_l1_out_channel, self.conv_l2_out_channel, kernel_size=self.conv_l2_kernel_size, padding=(1,1))

        self.out_height, self.output_width = self.calculate_conv2d_out_length(self.conv1, self.nn_input_length, self.conv_l1_in_width)
        self.padding1 = nn.ZeroPad2d((0, 0, (self.nn_input_length - self.out_height), (self.output_width-self.conv_l1_in_width)))
        self.out_height, self.output_width = self.calculate_conv2d_out_length(self.conv2, self.nn_input_length, self.conv_l1_in_width)
        self.padding2 = nn.ZeroPad2d(
            (0, 0, (self.nn_input_length - self.out_height), (self.output_width - self.conv_l1_in_width)))

        output_feature_size = self.calculate_flaten_out_length(self.conv_l2_out_channel, self.output_width)
        config['DecoderLSTM_input_size'] = output_feature_size

    def calculate_flaten_out_length(self, channel_size, input_length):
        return int(channel_size*input_length)

    def calculate_maxpool1d_out_length(self, maxpool_layer, input_length):
        maxpool_output_length = 1 + ((input_length + (2*maxpool_layer.padding) - maxpool_layer.dilation-(maxpool_layer.kernel_size-1))/maxpool_layer.stride)
        return int(maxpool_output_length)

    def calculate_conv2d_out_length(self, conv_layer, input_height, input_width):
        '''
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        :param conv_layer:
        :return:
        '''
        conv1d_output_height = 1 + ((input_height + (2*conv_layer.padding[0])- conv_layer.dilation[0]-(conv_layer.kernel_size[0]-1))/conv_layer.stride[0])
        conv1d_output_width = 1 + ((input_width + (2 * conv_layer.padding[1]) - conv_layer.dilation[1] - (
                    conv_layer.kernel_size[1] - 1)) / conv_layer.stride[1])
        return int(conv1d_output_height), int(conv1d_output_width)


    def forward(self, x):
        """Extract feature vectors from input signal."""
        x = torch.unsqueeze(x, 1)
        out = self.conv1(x)
        out = self.padding1(out)
        out = F.relu(out)
        out = nn.Dropout(p=self.conv_l1_dropout)(out)
        out = self.conv2(out)
        out = self.padding2(out)
        out = F.relu(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.flatten(out, start_dim=2)
        out = nn.Dropout(p=self.conv_l2_dropout)(out)
        return out


class DecoderFC(nn.Module):
    def __init__(self, config):
        """Set the hyper-parameters and build the layers."""
        super(DecoderFC, self).__init__()
        self.n_output = len(config['selected_opensim_labels'])
        self.fc2 = nn.Linear(230400, self.n_output)

    def forward(self, x):
        if x.ndim>2:
            x = torch.flatten(x, 1)
        out = F.softmax(self.fc2(x))
        return out


class DecoderLSTM(nn.Module):
    def __init__(self, config):
        """Set the hyper-parameters and build the layers."""
        super(DecoderLSTM, self).__init__()
        self.lstm_input_size = config['DecoderLSTM_input_size']
        self.lstm_hidden_size = config['DecoderLSTM_hidden_size']
        self.lstm_num_layers = config['DecoderLSTM_num_layers']
        dropout_p = config['DecoderLSTM_dropout_p']
        self.dropout = nn.Dropout(p=dropout_p)

        self.lstm1 = nn.LSTM(input_size=self.lstm_input_size,
                             hidden_size=self.lstm_hidden_size,
                             num_layers=self.lstm_num_layers,
                             batch_first=True)

    def forward(self, x):
        """Decode signal feature vectors and generates time series."""
        if x.ndim <3:
            x = x.unsqueeze_(dim=2)
        out, (h, c) = self.lstm1(x)
        out = self.dropout(out)
        return out


class Hernandez2021CNNLSTM(nn.Module):
    def __init__(self, config):
        super(Hernandez2021CNNLSTM, self).__init__()
        self.classification = config['classification']
        self.n_input_channel = len(config['selected_sensors'])*6
        self.n_output = len(config['selected_opensim_labels'])
        self.sequence_length = config['target_padding_length']

        config['EncoderCNN_conv_l1_in_channel'] = 1
        config['EncoderCNN_conv_l1_in_width'] = self.n_input_channel
        config['EncoderCNN_cnn_fc1_out_feature'] = self.sequence_length
        self.cnn = EncoderCNN(config)

        self.lstm_hidden_size = config['DecoderLSTM_hidden_size']
        self.lstm = DecoderLSTM(config)
        self.fc1 = nn.Linear(self.lstm_hidden_size, self.n_output)
        self.classifier = DecoderFC(config)


    def forward(self, x):
        c_out = self.cnn(x)
        l_out = self.lstm(c_out)
        regression_out = self.fc1(l_out)
        return regression_out