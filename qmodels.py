import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd.variable import Variable

class QGenerator(nn.Module):
    def __init__(self, z_dim, output_size, hidRNN=100, nlayers=1 ):
        super(QGenerator, self).__init__()
        self.hidR       = hidRNN
        self.z_dim      = z_dim
        self.output_size = output_size
        self.nlayers    = nlayers
        self.LSTM       = nn.LSTM(self.z_dim, self.hidR, self.nlayers, batch_first=True)
        # self.GRU1       = nn.GRU(self.hidC, self.hidR)
        self.fc         = nn.Linear(self.hidR, self.output_size)
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
    def __init__(self, nfeatures, hidRNN=100, nlayers=1 ):
        super(QDiscriminator, self).__init__()
        self.output_size= nfeatures
        self.nfeatures  = nfeatures
        self.hidR       = hidRNN
        self.nlayers    = nlayers
        self.LSTM       = nn.LSTM(self.nfeatures, self.hidR, self.nlayers, batch_first=True)
        # self.GRU1       = nn.GRU(self.hidC, self.hidR)
        self.fc         = nn.Linear(self.hidR, self.output_size)
        self.sigmoid    = nn.Sigmoid()
        print('\n Created Discriminator Class:\n ' + str(self))

    def forward(self, x):
        out,h = self.LSTM(x)
        out = self.fc(out)
        # out = self.sigmoid(out)
        return out


class GAN():
    def __init__(self, a_generator:nn.Module, a_discriminator:nn.Module, lr=0.05, config=None):
        self.generator      = a_generator
        self.discriminator  = a_discriminator
        self.lr = lr
        self.g_optimizer    = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5,0.999))
        self.d_optimizer    = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5,0.999))
        # self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=self.lr)
        # self.loss = nn.BCEWithLogitsLoss()
        # self.loss = nn.BCELoss(reduction='mean')
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.config = config

    def train_discriminator(self, real_data, fake_data):
        self.d_optimizer.zero_grad()
        label_as_real = Variable(torch.ones_like(real_data),requires_grad=False)
        label_as_fake = Variable(torch.zeros_like(real_data),requires_grad=False)

        # Measure discriminator's ability to classify real from generated samples
        real_logits = self.discriminator(real_data)
        fake_logits = self.discriminator(fake_data.detach())

        real_loss = self.loss(real_logits, label_as_real)
        fake_loss = self.loss(fake_logits, label_as_fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        self.d_optimizer.step()

        return d_loss

    def train_discriminator2(self, real_data, fake_data):
        self.d_optimizer.zero_grad()

        real_logits = self.discriminator(real_data)
        label_as_real = Variable(torch.ones_like(real_data),requires_grad=False)
        real_loss = self.loss(real_logits, label_as_real)
        # err_real.backward()

        fake_logits = self.discriminator(fake_data)
        label_as_fake = Variable(torch.zeros_like(real_data),requires_grad=False)
        fake_loss = self.loss(fake_logits, label_as_fake)
        # err_fake.backward()
        d_loss = ( real_loss + fake_loss ) / 2
        d_loss.backward()

        self.d_optimizer.step()

        return d_loss, real_logits, fake_logits

    def train_generator(self, fake_data):

        self.g_optimizer.zero_grad()

        label_as_real = Variable(torch.ones_like(fake_data))
        gen_logits = self.discriminator(fake_data)
        g_loss = self.loss(gen_logits, label_as_real)
        g_loss.backward()

        self.g_optimizer.step()
        return g_loss

    def save(self, path, epoch=0):
        torch.save({'generator_state_dict':      self.generator.state_dict(),
                   'discriminator_state_dict':  self.discriminator.state_dict(),
                   'generator_optimizer_state_dict': self.g_optimizer.state_dict(),
                   'discriminator_optimizer_state_dict': self.d_optimizer.state_dict(),
                   'config':self.config,
                   'epoch':epoch
                   }, path)

    def load(self, path):
        f = torch.load(path)
        self.generator.load_state_dict(f['generator_state_dict'])
        self.discriminator.load_state_dict(f['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(f['generator_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(f['discriminator_optimizer_state_dict'])

class GAN_factory:
    @staticmethod
    def default_gan(latent_dim = 10, nfeatures=1 , gen_hidRNN=100, gen_memory_layers=1,
                    dis_hidRNN=100, dis_memory_layers=1, use_cuda=False, lr=0.05, config=None):
        device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        generator = QGenerator(z_dim=latent_dim, output_size=nfeatures, hidRNN=gen_hidRNN, nlayers=gen_memory_layers).to(device)
        discriminator = QDiscriminator(nfeatures=nfeatures, hidRNN=dis_hidRNN, nlayers=dis_memory_layers).to(device)
        # generator = generator.float()
        # discriminator = discriminator.float()
        gan = GAN(generator, discriminator, lr, config=config)
        gan.loss.to(device)
        # for name, param in generator.named_parameters():
        #     print('\n generator parameter: '+name)
        return gan

    @staticmethod
    def model_from_config(cfg):
        # only default gan supported at this time
        is_default = 'RNN' in cfg['generator_type'] and 'RNN' in cfg['discriminator_type'] \
                     and \
                     'LSTM' in cfg['generator_RNN_cell'] and 'LSTM' in cfg['discriminator_RNN_cell']
        if not is_default:
            raise Exception('Only default GAN is supported at this time, please choose RNN and LSTM as type and cell for G and D')

        return GAN_factory.default_gan(latent_dim=cfg['latent_dimension'], nfeatures=cfg['nfeatures'],
                                       gen_hidRNN=cfg['generator_hidden_units'], gen_memory_layers=cfg['generator_RNN_layers'],
                                       dis_hidRNN=cfg['discriminator_hidden_units'], dis_memory_layers=cfg['discriminator_RNN_layers'],
                                       use_cuda=cfg['use_cuda'], lr=cfg['lr'], config=cfg)

    @staticmethod
    def model_from_checkpoint(path):
        f=torch.load(path)
        the_gan = GAN_factory.model_from_config(f['config'])
        the_gan.load(path)
        return the_gan, f['config'], f['epoch']

def test_QGenerator():
    zdim        = 5
    seq_length  = 200
    batch       = 10
    hidden_sz   = 100
    ndims = 2
    # input = torch.randn(batch, seq_length, zdim)
    z = torch.ones(batch, seq_length, zdim)
    gen = QGenerator(z_dim=zdim, output_size= ndims)
    x_hat = gen(z)
    k=0


def test_QGDiscriminator():
    seq_length  = 200
    batch       = 10
    hidden_sz   = 100
    ndims = 2
    # input = torch.randn(batch, seq_length, zdim)
    x = torch.ones(batch, seq_length, ndims)
    dis = QDiscriminator(nfeatures=ndims)
    x_hat = dis(x)
    k=0

