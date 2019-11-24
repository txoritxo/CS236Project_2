import qmodels
import qconfig
import os.path
import sys
import numpy as np
import torch
import Logger
from Logger import Logger
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
from time import time
from torch.utils.data.sampler import SubsetRandomSampler

qseed = 125

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

def noise(size, device):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size)).to(device)
    return n

def load_dataset(cfg):
    filename = os.path.join('dataset', cfg['dataset'])
    print('\n Loading dataset file ' + filename)
    data = np.load(filename)
    if data is None:
        raise Exception('Failed to open dataset: ' + filename)

    np.random.seed(qseed)
    np.random.shuffle(data)

    dataset_size = len(data)
    # indices = list(range(dataset_size))
    # np.random.shuffle(indices)
    split = int(np.floor(cfg['split_train'] * dataset_size))
    # train_indices, val_indices = indices[:split], indices[split:]

    nsamples  = data.shape[0]
    lsequence = data.shape[1]
    nfeatures = data.shape[2]
    print('\ndataset loaded, contains {} samples, sequence length is {} and features are {}'.format(
        dataset_size, lsequence, nfeatures))

    cfg['nsamples']  = nsamples
    cfg['lsequence'] = lsequence
    cfg['nfeatures'] = nfeatures
    train_data = data[:split,]
    val_data   = data[split:]

    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'],
                                                   drop_last=True)
    print('\n \ndataset has been split {} % train ({} samples) and {} % validation ({} samples)\n\n'.format(
          cfg['split_train']*100, split, 100-cfg['split_train']*100, dataset_size-split))

    return train_dataloader, torch.from_numpy(val_data)

def resume_training(checkpoint_file):
    try:
        the_gan, cfg, epoch = qmodels.GAN_factory.model_from_checkpoint(checkpoint_file)
        return the_gan, cfg, epoch
    except:
        print('\n error loading model from checkpoint file ' + checkpoint_file)
        sys.exit()


def run_experiment(filename):
    the_gan = None
    cfg = None
    epoch0 = 0

    if filename.endswith('.ini'):
        cfg=qconfig.parse_config_file(filename)
        train_dataloader, _ = load_dataset(cfg)
        the_gan = qmodels.GAN_factory.model_from_config(cfg)
    elif filename.endswith('.pth'):
        the_gan, cfg, epoch0 = resume_training(filename)
        train_dataloader, _ = load_dataset(cfg)
    else:
        raise Exception('file ' + filename + ' not supported, use a .pth or .cfg file')

    directory=os.path.join('experiments', cfg['id'])
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['use_cuda'] else 'cpu')
    num_batches = len(train_dataloader)
    logger = Logger(model_name=cfg['id'], dir='logs/'+cfg['id'], model=the_gan, epoch=epoch0)

    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
    nepochs = cfg['nepochs']

    t0 = time()
    for epoch in range(epoch0, nepochs):
        for batch,(batch_data) in enumerate(train_dataloader):
            # Configure input
            real_data = Variable(batch_data.type(Tensor)).to(device)
            for g_loop in range(cfg['G_loops']):
                # -----------------
                #  Train Generator
                # -----------------
                # # Sample noise as generator input
                z = noise((cfg['batch_size'], cfg['lsequence'], cfg['latent_dimension']), device)
                # # Generate a batch of images
                gen_data = the_gan.generator(z)
                g_loss = the_gan.train_generator(gen_data)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for d_loop in range(cfg['D_loops']):
                z = noise((cfg['batch_size'], cfg['lsequence'], cfg['latent_dimension']), device)
                gen_data = the_gan.generator(z)
                d_loss = the_gan.train_discriminator(real_data, gen_data)

        t = time() - t0
        logger.on_epoch(t, epoch, g_loss, d_loss, gen_data, real_data)

# run_experiment('config/template.ini')
run_experiment('config/revisit_milestone.ini')

# run_experiment("./logs/FirstTest/checkpoint/FirstTest-e0670.pth")
