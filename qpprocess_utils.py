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

    pool = mp.Pool(ncpus)
    result_objects = [pool.apply_async(compute_distance_for_sample, args=(sample_id,fake_data,real_data,nsamples)) for sample_id in range(nsamples)]
    results = [r.get()[1] for r in result_objects]
    pool.close()
    pool.join()
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


# generate_plots_for_model('./logs/GW60KSE3D_GRU1/checkpoint/GW60KSE3D_GRU1-e0000.pth', numsamples=100)
# generate_dtw_for_epoch_range('./config/GW60KSE3D_GRU01.ini', start_epoch=0, end_epoch=5, skip_epoch=1, nsamples=20)
parser = argparse.ArgumentParser(description='')
parser.add_argument('--generate_dtw', help='tbd', action='store_true')
parser.add_argument('--generate_plots', help='tbd', action='store_false')
parser.add_argument('--input_file', help='tbd', default="config/template.ini")
parser.add_argument('--start_epoch', help='tbd', default=0)
parser.add_argument('--end_epoch', help='tbd', default=10000)
parser.add_argument('--nsamples', help='tbd', default=100)

args = parser.parse_args()
if args.generate_dtw:
    generate_dtw_for_epoch_range(args.input_file, start_epoch=int(args.start_epoch), end_epoch=int(args.end_epoch),
                                 skip_epoch=1, nsamples=int(args.nsamples))
elif args.generate_plots:
    generate_plots_for_model(args.input_file, numsamples=int(args.nsamples))
else:
    print('\nPlease select --generate_dtw or --generate_plots')
