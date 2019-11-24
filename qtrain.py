import torch
import torch.utils
from torch.autograd.variable import Variable
import numpy as np
import qmodels
from Logger import Logger
import matplotlib.pyplot as plt
from time import time

def noise(size, device):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size)).to(device)
    return n

def generate_sine_data(nsamples=20000, seq_length=200, nsignals=1, freq_low=1, freq_high=5,
                       amplitude_low=0.1, amplitude_high=0.9):
    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(nsamples):
        signals = []
        for i in range(nsignals):
            f = np.random.uniform(low=freq_high, high=freq_low)  # frequency
            A = np.random.uniform(low=amplitude_high, high=amplitude_low)  # amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            signals.append(A * np.sin(2 * np.pi * f * ix / float(seq_length) + offset))
        samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples, dtype=np.float32)
    return torch.from_numpy(samples)

def plot_data(data, name='temp'):
    fig = plt.figure()
    ax = plt.axes()
    batches = min(10, len(data))
    xpoints = np.arange(len(data[0]))
    for i in range(batches):
        ypoints = data[i,:,0].cpu().numpy()
        ax.plot(xpoints, ypoints)
    fig.savefig('./sandbox/'+name + '.png')
    plt.close()
    k=0

def train2(data):
    nepochs = 100
    batch_sz = 100
    sequence = data.shape[1]
    nfeatures = 1
    latent_dim = 20
    g_hidden_dim = 50
    d_hidden_dim = 50
    g_nlayers = 1
    d_nlayers = 1
    gen_loops = 1
    dis_loops = 1
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    t0 = time()
    nsamples = data.shape[0]
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    qdata_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_sz, shuffle=True)

    the_gan = qmodels.GAN_factory.default_gan(latent_dim=latent_dim, nfeatures=nfeatures, gen_hidRNN=g_hidden_dim, gen_memory_layers=g_nlayers,
                                          dis_hidRNN=d_hidden_dim, dis_memory_layers=d_nlayers, use_cuda=use_cuda, lr=lr)

    num_batches = len(qdata_loader)
    logger = Logger(model_name='TESTGANNN', dir='TEMP', model=the_gan)
    logger.qdisplay_header()


    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor

    for epoch in range(nepochs):

        for batch,(batch_data) in enumerate(qdata_loader):
            # Configure input
            real_imgs = Variable(batch_data.type(Tensor)).to(device)

            # -----------------
            #  Train Generator
            # -----------------
            # # Sample noise as generator input
            z = noise((batch_sz, sequence, latent_dim), device)
            # # Generate a batch of images
            gen_data = the_gan.generator(z)
            g_loss = the_gan.train_generator(gen_data)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            d_loss = the_gan.train_discriminator(real_imgs, gen_data)

            if batch % 20 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, nepochs, batch, len(qdata_loader), d_loss.item(), g_loss.item())
                )
                # plot_data(real_imgs, 'real-e' + str(epoch) + '_b' + str(i))
                plot_data(gen_data.detach(), 'fake-e' + str(epoch) + '_b' + str(batch))

            batches_done = epoch * len(qdata_loader) + batch
        # name = 'mymodel_e'+str(epoch)+'.pth'
        # the_gan.save_model('checkpoints/'+name)

def train(data):
    logger = Logger(model_name='TESTGAN', data_name='TEST')
    nepochs = 100
    batch_sz = 100
    sequence = data.shape[1]
    nfeatures = 1
    latent_dim = 20
    g_hidden_dim = 50
    d_hidden_dim = 50
    g_nlayers = 1
    d_nlayers = 1
    gen_loops = 1
    dis_loops = 1
    learning_rate = 0.0002
    t0 = time()
    nsamples = data.shape[0]
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    qdata_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_sz, shuffle=True)

    the_gan = qmodels.GAN_factory.default_gan(latent_dim=latent_dim, nfeatures=nfeatures, gen_hidRNN=g_hidden_dim, gen_memory_layers=g_nlayers,
                                          dis_hidRNN=d_hidden_dim, dis_memory_layers=d_nlayers, use_cuda=use_cuda, lr=learning_rate)

    num_batches = len(qdata_loader)
    logger.qdisplay_header()

    for epoch in range(nepochs):
        g_loss_acum = 0
        d_loss_acum = 0

        for nbatch,(real_batch) in enumerate(qdata_loader):
            real_data = Variable(real_batch).to(device)
            # 1.- Train Discriminator
            fake_data = the_gan.generator(noise((batch_sz, sequence, latent_dim), device)).detach().to(device)

            for d in range(dis_loops):
                d_error, d_pred_real, d_pred_fake = the_gan.train_discriminator(real_data, fake_data)

            #2.- Train Generator
            for g in range(gen_loops):
                # fake_data = the_gan.generator(noise((batch_sz, sequence, latent_dim), device)).detach()
                g_error = the_gan.train_generator(fake_data)

            g_loss_acum += g_error
            d_loss_acum += d_error
            if (nbatch) % 20 == 0:
                t=time()-t0
                # plot_data(real_data, 'real-e'+str(epoch)+'_b'+str(nbatch))
                plot_data(fake_data, 'fake-e'+str(epoch)+'_b'+str(nbatch))
                logger.qdisplay_status(t, epoch, nepochs, nbatch, num_batches,d_error, g_error,
                                       g_loss_acum/(nbatch+1), d_loss_acum/(nbatch+1))


def qsine():
    nsamples   = 10000
    seq_length = 50
    data = generate_sine_data(nsamples, seq_length)
    train2(data)


qsine()