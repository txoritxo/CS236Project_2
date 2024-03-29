import qgan_factory
import pickle
import os.path
import numpy as np
from torch.autograd.variable import Variable
import torch
import Logger
from Logger import Logger
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import qconfig
from time import time
import multiprocessing as mp
import argparse
from matplotlib import pyplot as plt
import pandas as pd

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_subdirs(subdirs):
        if not os.path.exists(subdirs):
            os.makedirs(subdirs)

def compute_dtw(a, b):
    distance, path = fastdtw(a, b, dist=euclidean)
    return distance, path

def get_checkpoint_path(identifier, epoch):
    checkpoint_name='{}-e{:04d}.pth'.format(identifier, epoch)
    checkpoint_path=os.path.join('./logs', identifier,'checkpoint',checkpoint_name)
    return checkpoint_path

def compute_distance_for_sample(sample_id,fake_data,real_data,nsamples):
    print('\nComputing distance for sample!!')
    distance=0
    x=fake_data[sample_id,]
    for j in range(nsamples):
        y=real_data[j,]
        cur_distance,_ = compute_dtw(x,y)
        distance+=cur_distance
    return distance

def generate_dtw_for_epoch_parallel(identifier, epoch, real_data, nsamples=None, device='cpu'):
    ncpus=4
    max_cpus = mp.cpu_count()
    ncpus=min(ncpus,max_cpus)

    checkpoint_path=get_checkpoint_path(identifier, epoch)
    the_gan, cfg, epoch_saved = qgan_factory.GAN_factory.model_from_checkpoint(checkpoint_path)

    if epoch_saved != epoch:
        print('\n number of epochs is different')
        raise Exception('\nNumber of epochs is different ')

    z = noise((nsamples, cfg['lsequence'], cfg['latent_dimension']), device)
    fake_data = the_gan.generator(z)[:,:,0:1]
    fake_data = fake_data.detach().cpu().numpy()
    total_distance=0
    with mp.Pool(processes=ncpus) as pool:
        result_objects = [pool.apply_async(compute_distance_for_sample, args=(sample_id,fake_data,real_data,nsamples)) for sample_id in range(nsamples)]
        results = [r.get()[1] for r in result_objects]
        pool.close()
        pool.join()

    print('\n I should be computing the mean here! **************************')
    mean = total_distance/nsamples**2
    return mean

def generate_dtw_for_epoch(identifier, epoch, real_data, nsamples=None, device='cpu'):
    checkpoint_path=get_checkpoint_path(identifier, epoch)
    the_gan, cfg, epoch_saved = qgan_factory.GAN_factory.model_from_checkpoint(checkpoint_path)

    if epoch_saved != epoch:
        print('\n number of epochs is different')
        raise Exception('\nNumber of epochs is different ')

    z = noise((nsamples, cfg['lsequence'], cfg['latent_dimension']), device)
    fake_data = the_gan.generator(z)[:,:,0:1]
    fake_data = fake_data.detach().cpu().numpy()
    total_distance=0
    for i in range(nsamples):
        x=fake_data[i,]
        for j in range(nsamples):
            y=real_data[j,]
            cur_distance, _ = compute_dtw(x,y)
        # mpd = mmd.median_pairwise_distance(X=x, Y=y)
            total_distance += cur_distance
        # print('\n current_distance is {:f}', cur_distance)

    mean = total_distance/nsamples**2
    return mean


def generate_dtw_for_epoch_range(config_file, start_epoch=1, end_epoch=None, skip_epoch=1, nsamples=20):
    cfg = qconfig.parse_config_file(config_file)
    dataset = load_dataset(cfg['dataset'])
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['use_cuda'] else 'cpu')
    np.random.shuffle(dataset)
    real_samples_np=dataset[0:nsamples,:,0:1]
    filename = 'DTW_' + cfg['id'] + '.csv'
    filedir = os.path.join('./logs', cfg['id'], 'results')
    create_subdirs(filedir)
    filepath = os.path.join(filedir, filename)
    f = open(filepath, 'w')
    t0 = time()
    for epoch in range(start_epoch, end_epoch+skip_epoch, skip_epoch):
        # metric = generate_dtw_for_epoch_parallel(cfg['id'], epoch, real_samples_np, nsamples, device=device)
        try:
            metric = generate_dtw_for_epoch(cfg['id'], epoch, real_samples_np, nsamples, device=device)
            #metric = generate_dtw_for_epoch_parallel(cfg['id'], epoch, real_samples_np, nsamples, device=device)
        except:
            print('Exception raised while trying to compute DTW for model ' + cfg['id']
                  + ' in epoch '+ str(epoch) + '. Skipping to next epoch')
            continue
        ## metric = generate_mmd2_for_epoch(identifier, epoch, real_samples, nsamples)
        t=time()-t0
        print('\n{:6.0f} {:04d} , {:f}'.format(t, epoch, metric))
        f.write('\n{:6.0f}, {:d}, {:f}'.format(t,epoch, metric))
        f.flush()
    f.close()

def noise(size, device):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size)).to(device)
    return n

def load_dataset(filename):
    filename = os.path.join('dataset', filename)
    print('\n Loading dataset file ' + filename)
    data=None
    if filename.endswith('.npy'):
        data = np.load(filename)
    elif filename.endswith('.pickle'):
        f = pickle.load(open(filename,'rb'))
        data = f['dataset']
    else:
        raise Exception('\n dataset file has to be either .npy or .pickle')
    if data is None:
        raise Exception('Failed to open dataset: ' + filename)

    return data

def load_model_and_dataset(filename):
    if not filename.endswith('pth'):
        raise Exception('model shall be a pth file')

    the_gan, cfg, epoch = qgan_factory.GAN_factory.model_from_checkpoint(filename)
    nfeatures = cfg['nfeatures']
    data = load_dataset(cfg['dataset'])
    data = data[:,:,0:nfeatures]

    return the_gan, data, cfg, epoch

def plot_DTW(filename='results.csv'):
    width=0.3
    df = pd.read_csv(filename)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(9,3)
    color = 'black'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(df['epoch'], df['D_loss'], color='blue',  linewidth=width, label='$D_{loss}$')
    ax1.plot(df['epoch'], df['G_loss'], color='green', linewidth=width, label='$G_{loss}$')
    # ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    ax2.set_ylabel('dtw', color=color)  # we already handled the x-label with ax1
    ax2.plot(df['epoch'], df['mmd2'], color='red', linewidth=width,label='DTW')
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.legend()
    ax2.legend()
    plt.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig('results.png')
    plt.close()


def generate_real_plots(filename, numsamples=10):
    data = load_dataset(filename)
    np.random.shuffle(data)
    real_samples_np=data[0:numsamples,:,0:1]
    fig = plt.figure()
    ax = plt.axes()
    plt.grid(b=True, color='grey', linestyle=':', linewidth=0.5)
    for i in range(numsamples):
        ypoints = data[i, :, 0]
        xpoints = data[i, :, 1]
        ax.plot(xpoints, ypoints, color='black', linewidth=0.05)
    plt.axis('equal')
    plt.xlabel('normalized longitude')
    plt.ylabel('normalized latitude')
    plt.title('2D Trajectory')
    fig.savefig('./real_trajectories_xy.png')
    plt.close()

    fig = plt.figure()
    ax = plt.axes()
    plt.grid(b=True, color='grey', linestyle=':', linewidth=0.5)
    for i in range(numsamples):
        zpoints = data[i, :, 2]
        ax.plot(zpoints, color='black', linewidth=0.05)
    plt.xlabel('trajectory sample')
    plt.ylabel('normalized altitude')
    plt.title('Altitude profile')
    fig.savefig('./real_trajectories_z.png')
    plt.close()

def plot_real_vs_gen(model, data, cfg, epoch=0, numsamples=10, title=None, force2D=None):
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['use_cuda'] else 'cpu')
    z = noise((numsamples, cfg['lsequence'], cfg['latent_dimension']), device)
    fake_data = model.generator(z)
    real_data = torch.from_numpy(data[0:numsamples])
    dir = os.path.join('./logs', cfg['id'])
    logger = Logger(model_name = cfg['id'], dir=dir, model=model, epoch=epoch, post_process=True)
    logger.plot(epoch, fake_data, real_data, numsamples, title=title, force2D=force2D)

def generate_plots_for_model(filename, numsamples=100, title=None, force2D=None):
    the_model, data, cfg, epoch = load_model_and_dataset(filename)
    plot_real_vs_gen(the_model, data, cfg, epoch, numsamples, title=title, force2D=force2D)


# generate_plots_for_model('./logs/GW60KSE3D_GRU1/checkpoint/GW60KSE3D_GRU1-e0000.pth', numsamples=100)
# generate_dtw_for_epoch_range('./config/GW60KSE3D_GRU01.ini', start_epoch=0, end_epoch=5, skip_epoch=1, nsamples=20)
parser = argparse.ArgumentParser(description='')
parser.add_argument('--generate_dtw', help='tbd',action='store_true')
parser.add_argument('--force2D', help='tbd',action='store_true')
parser.add_argument('--generate_plots', help='tbd',action='store_true')
parser.add_argument('--generate_real_plots', help='tbd',action='store_true')
parser.add_argument('--input_file', help='tbd', default="config/template.ini")
parser.add_argument('--start_epoch', help='tbd', default=0)
parser.add_argument('--end_epoch', help='tbd', default=10000)
parser.add_argument('--nsamples', help='tbd', default=100)
parser.add_argument('--title', help='tbd', default=None)

args = parser.parse_args()
force2D=True if args.force2D else False

if args.generate_dtw:
    generate_dtw_for_epoch_range(args.input_file, start_epoch=int(args.start_epoch), end_epoch=int(args.end_epoch),
                                 skip_epoch=1, nsamples=int(args.nsamples))
elif args.generate_plots:
    generate_plots_for_model(args.input_file, numsamples=int(args.nsamples), title=args.title, force2D=force2D)
elif args.generate_real_plots:
    generate_real_plots(args.input_file, numsamples=int(args.nsamples), title=args.title)
else:
    print('\nPlease select --generate_dtw or --generate_plots')


#python pprocess_utils.py --generate_plots --input_file ./logs/GW60KSE3D_MMD_GRU02/checkpoint/GW60KSE3D_MMD_GRU02-e0199.pth --nsamples 75 --title "2-layer GRU with Minibatch Discrimination layer"