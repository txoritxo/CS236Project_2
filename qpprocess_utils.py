import qgan_factory
import pickle
import os.path
import numpy as np
from torch.autograd.variable import Variable
import torch
from Logger import Logger

def noise(size, device):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size)).to(device)
    return n


def load_model_and_dataset(filename):
    if not filename.endswith('pth'):
        raise Exception('model shall be a pth file')

    the_gan, cfg, epoch = qgan_factory.GAN_factory.model_from_checkpoint(filename)
    filename = os.path.join('dataset', cfg['dataset'])
    print('\n Loading dataset file ' + filename)
    data=None
    nfeatures = cfg['nfeatures']
    if filename.endswith('.npy'):
        data = np.load(filename)
    elif filename.endswith('.pickle'):
        f = pickle.load(open(filename,'rb'))
        data = f['dataset']
    else:
        raise Exception('\n dataset file has to be either .npy or .pickle')
    if data is None:
        raise Exception('Failed to open dataset: ' + filename)

    data = data[:,:,0:nfeatures]

    return the_gan, data, cfg, epoch


def plot_real_vs_gen(model, data, cfg, epoch=0, numsamples=10):
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['use_cuda'] else 'cpu')
    z = noise((numsamples, cfg['lsequence'], cfg['latent_dimension']), device)
    fake_data = model.generator(z)
    real_data = torch.from_numpy(data[0:numsamples])
    dir = os.path.join('./logs', cfg['id'])
    logger = Logger(model_name = cfg['id'], dir=dir, model=model, epoch=epoch, post_process=True)
    logger.plot(epoch, fake_data, real_data, numsamples)

def generate_plots_for_model(filename, numsamples=100):
    the_model, data, cfg, epoch = load_model_and_dataset(filename)
    plot_real_vs_gen(the_model, data, cfg, epoch, numsamples)


generate_plots_for_model('./logs/GW60KSE3D_GRU1/checkpoint/GW60KSE3D_GRU1-e0000.pth', numsamples=100)