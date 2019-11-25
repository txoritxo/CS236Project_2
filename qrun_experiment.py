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
import argparse

qseed = 125

class GenDisLoop:
    max_D_loops = 5
    max_G_loops = 5
    switch_epoch = 5
    def __init__(self, is_adaptative, G_loops, D_loops):
        self.is_adaptative = is_adaptative
        self.D_loops = GenDisLoop.max_D_loops if  is_adaptative else D_loops
        self.G_loops = 1 if is_adaptative else G_loops

    def __get_nloops(self, epoch, current_val, max_val):
        if epoch % GenDisLoop.switch_epoch == 0 and self.is_adaptative:
            current_val = 1 if current_val == max_val else max_val
        return current_val

    def get_D_loops(self, epoch):
        self.D_loops = self.__get_nloops(epoch, self.D_loops, self.max_D_loops)
        return self.D_loops

    def get_G_loops(self, epoch):
        self.G_loops = self.__get_nloops(epoch, self.G_loops, self.max_G_loops)
        return self.G_loops


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

def copy_config_file_to_logs_folder(filename, id):
    import shutil
    target_dir = os.path.join('logs', id)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    shutil.copy2(filename, target_dir)

def run_experiment(filename, is_adaptative):
    the_gan = None
    cfg = None
    epoch0 = 0

    if filename.endswith('.ini'):
        cfg=qconfig.parse_config_file(filename)
        train_dataloader, _ = load_dataset(cfg)
        the_gan = qmodels.GAN_factory.model_from_config(cfg)
        copy_config_file_to_logs_folder(filename, cfg['id'])

    elif filename.endswith('.pth'):
        the_gan, cfg, epoch0 = resume_training(filename)
        train_dataloader, _ = load_dataset(cfg)
    else:
        raise Exception('file ' + filename + ' not supported, use a .pth or .ini file')

    directory=os.path.join('experiments', cfg['id'])
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['use_cuda'] else 'cpu')
    num_batches = len(train_dataloader)
    logger = Logger(model_name=cfg['id'], dir='logs/'+cfg['id'], model=the_gan, epoch=epoch0)

    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
    nepochs = cfg['nepochs']

    nloop = GenDisLoop(is_adaptative, cfg['G_loops'], cfg['D_loops'])

    t0 = time()

    for epoch in range(epoch0, nepochs):
        g_loops = nloop.get_G_loops(epoch+1)
        d_loops = nloop.get_D_loops(epoch+1)
        # print('\nEpoch {} training Generator {} times and Discriminator {} times\n'.format(epoch, g_loops, d_loops))
        for batch,(batch_data) in enumerate(train_dataloader):
            # Configure input
            real_data = Variable(batch_data.type(Tensor)).to(device)
            for g_loop in range(g_loops):
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
            for d_loop in range(d_loops):
                z = noise((cfg['batch_size'], cfg['lsequence'], cfg['latent_dimension']), device)
                gen_data = the_gan.generator(z)
                d_loss = the_gan.train_discriminator(real_data, gen_data)

        t = time() - t0
        logger.on_epoch(t, epoch, g_loss, d_loss, gen_data, real_data)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', help='ini configuration file that defines the experiment', default="config/template.ini")
parser.add_argument("--adaptative", help="changes G and D training loops dynamically", action='store_true')
args = parser.parse_args()
run_experiment(args.config_file, args.adaptative)
#run_experiment('config/revisit_milestone.ini')

# run_experiment("./logs/FirstTest/checkpoint/FirstTest-e0670.pth")
