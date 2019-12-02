import torch
import torch.nn as nn
import torchgan

class QGenerator(nn.Module):
    def __init__(self, z_dim, output_size, hidRNN=100, nlayers=1, bidirectional=False, cell_type='LSTM', dropout=0):
        super(QGenerator, self).__init__()
        num_directions = 2 if bidirectional else 1
        self.hidR       = hidRNN
        self.z_dim      = z_dim
        self.output_size = output_size
        self.nlayers    = nlayers
        if cell_type in "LSTM":
            self.LSTM   = nn.LSTM(self.z_dim, self.hidR, self.nlayers, batch_first=True, bidirectional=bidirectional,dropout=dropout)
        elif cell_type in "GRU":
            self.LSTM = nn.GRU(self.z_dim, self.hidR, self.nlayers, batch_first=True, bidirectional=bidirectional,dropout=dropout)
        else:
            raise Exception('Cell Type {} not recognized building the Generator'.format(cell_type))

        # self.GRU1       = nn.GRU(self.hidC, self.hidR)
        self.fc         = nn.Linear(self.hidR*num_directions, self.output_size)
        self.tanh       = nn.Tanh()
        # self.init_lstm_bias()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print (name, param.data)
        print('\n Created Generator Class:\n ' + str(self))

    def forward(self,z):
        out,h = self.LSTM(z)
        out = self.fc(out)
        out = self.tanh(out)
        return out

    def init_lstm_bias(self):
        for name, param in self.LSTM.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0.0, std=1.0)
            if 'bias' in name:
                param.data.fill_(1)
                print('\n bias set to 1 in generator')

class QDiscriminator(nn.Module):
    def __init__(self, nfeatures, hidRNN=100, nlayers=1, bidirectional=False, cell_type='LSTM',dropout=0):
        super(QDiscriminator, self).__init__()
        num_directions = 2 if bidirectional else 1
        self.output_size= nfeatures
        self.nfeatures  = nfeatures
        self.hidR       = hidRNN
        self.nlayers    = nlayers
        if cell_type in "LSTM":
            self.LSTM   = nn.LSTM(self.nfeatures, self.hidR, self.nlayers, batch_first=True, bidirectional=bidirectional,dropout=dropout)
        elif cell_type in "GRU":
            self.LSTM= nn.GRU(self.nfeatures, self.hidR, self.nlayers, batch_first=True, bidirectional=bidirectional,dropout=dropout)
        else:
            raise Exception('Cell Type {} not recognized building the Discriminator'.format(cell_type))

        # self.GRU1       = nn.GRU(self.hidC, self.hidR)
        self.fc         = nn.Linear(self.hidR*num_directions, self.output_size)
        self.sigmoid    = nn.Sigmoid()
        print('\n Created Discriminator Class:\n ' + str(self))

    def forward(self, x):
        out,h = self.LSTM(x)
        out = self.fc(out)
        # out = self.sigmoid(out)
        return out


class QDiscriminator_mmd(nn.Module): #minibatch_discrimination layer
    def __init__(self, nfeatures=1, hidRNN=100, nlayers=1, bidirectional=False, cell_type='LSTM',dropout=0,
                 mmd_kernel_size=16, mmd_out_features=200):
        super(QDiscriminator_mmd, self).__init__()
        num_directions = 2 if bidirectional else 1
        self.output_size= nfeatures
        self.nfeatures  = nfeatures
        self.hidR       = hidRNN
        self.nlayers    = nlayers
        self.mmd_out_features = mmd_out_features
        self.mmd_kernel_size = mmd_kernel_size
        self.seq_length=200
        if cell_type in "LSTM":
            self.LSTM   = nn.LSTM(self.nfeatures, self.hidR, self.nlayers, batch_first=True, bidirectional=bidirectional,dropout=dropout)
        elif cell_type in "GRU":
            self.LSTM= nn.GRU(self.nfeatures, self.hidR, self.nlayers, batch_first=True, bidirectional=bidirectional,dropout=dropout)
        else:
            raise Exception('Cell Type {} not recognized building the Discriminator'.format(cell_type))

        # self.GRU1       = nn.GRU(self.hidC, self.hidR)
        self.fc     = nn.Linear(self.hidR*num_directions, self.output_size)
        self.mmd = torchgan.layers.MinibatchDiscrimination1d(self.seq_length * self.output_size,
                                                             self.mmd_out_features, self.mmd_kernel_size)
        self.fc2 = nn.Linear(self.seq_length*self.output_size+self.mmd_out_features, self.seq_length*self.output_size)
        print('\n Created Discriminator Class:\n ' + str(self))

    def forward(self, x):
        out,h = self.LSTM(x)
        out = self.fc(out)
        out = torch.flatten(out, start_dim=1)
        out = self.mmd(out)
        out = self.fc2(out)
        out = out.view(-1,self.seq_length,self.output_size)

        # out = self.sigmoid(out)
        return out

