import configparser
from pathlib import Path

def parse_config_file(filename):
    the_config = {}
    config = configparser.ConfigParser()
    config.read(filename)
    if 'experiment' in config.sections():
        the_config['id']      = config['experiment'].get('id', 'Unknown')
        the_config['dataset'] = config['experiment'].get('dataset', 'Unknown')
    else:
        raise Exception('configuration file ' + filename + ' does not have a [experiment] section')

    if 'model' in config.sections():
        the_config['latent_dimension'] = config['model'].getint('latent_dimension', 10)
        the_config['nfeatures'] = config['model'].getint('nfeatures', 2)
        the_config['generator_type'] = config['model'].get('generator_type', 'RNN')
        the_config['generator_RNN_cell'] = config['model'].get('generator_RNN_cell', 'LSTM')
        the_config['generator_hidden_units'] = config['model'].getint('generator_hidden_units', 10)
        the_config['generator_RNN_layers'] = config['model'].getint('generator_RNN_layers', 1)
        the_config['generator_RNN_layers'] = config['model'].getint('generator_RNN_layers', 1)
        the_config['generator_dropout'] = config['model'].getfloat('generator_dropout', 0)

        the_config['discriminator_type'] = config['model'].get('discriminator_type', 'RNN')
        the_config['discriminator_RNN_cell'] = config['model'].get('discriminator_RNN_cell', 'LSTM')
        the_config['discriminator_hidden_units'] = config['model'].getint('discriminator_hidden_units', 10)
        the_config['discriminator_RNN_layers'] = config['model'].getint('discriminator_RNN_layers', 1)
        the_config['discriminator_dropout'] = config['model'].getfloat('discriminator_dropout', 0)
    else:
        raise Exception('configuration file does not have a [model] section')

    if 'training' in config.sections():
        the_config['use_cuda']  = config['training'].getboolean('use_cuda', True)
        the_config['multi_gpu'] = config['training'].getboolean('multi_gpu', False)
        the_config['shuffle']   = config['training'].getboolean('shuffle', False)
        the_config['nepochs']   = config['training'].getint('epochs', 100)
        the_config['batch_size'] = config['training'].getint('batch_size', 10)
        the_config['D_loops']   = config['training'].getint('discriminator_loops_per_epoch', 1)
        the_config['G_loops']   = config['training'].getint('generator_loops_per_epoch', 1)
        the_config['lr']        = config['training'].getfloat('learning_rate', 0.0002)
        the_config['split_train'] = config['training'].getfloat('split_train', 0.8)
    else:
        raise Exception('configuration file does not have a [training] section')

    return the_config

def parse_dataset_config(filename):
    the_config = {}
    config = configparser.ConfigParser()
    config.read(filename)
    if 'dataset' in config.sections():
        the_config['id'] = config['dataset'].get('id', 'Unknown')
        the_config['source'] = Path(config['dataset'].get('source','./'))
        the_config['nfiles'] = config['dataset'].getint('nfiles', 1)
        the_config['nsamples'] = config['dataset'].getint('samples_per_flight',200)
        use_altitude =  config['dataset'].getboolean('use_altitude',False)
        the_config['nfeatures'] = 3 if use_altitude else 2
        the_config['resample_period'] = config['dataset'].getint('resample_period', 1)
        start_min_lon = config['dataset'].getfloat('start_min_lon', None)
        start_max_lon = config['dataset'].getfloat('start_max_lon', None)
        start_min_lat = config['dataset'].getfloat('start_min_lat', None)
        start_max_lat = config['dataset'].getfloat('start_max_lat', None)
        the_config['start_window'] = {'lat':(start_min_lat,start_max_lat), 'lon':(start_min_lon,start_max_lon)}
        end_min_lon = config['dataset'].getfloat('end_min_lon', None)
        end_max_lon = config['dataset'].getfloat('end_max_lon', None)
        end_min_lat = config['dataset'].getfloat('end_min_lat', None)
        end_max_lat = config['dataset'].getfloat('end_max_lat', None)
        the_config['end_window'] = {'lat':(end_min_lat,end_max_lat), 'lon':(end_min_lon,end_max_lon)}
        alt_min = config['dataset'].getfloat('alt_min', None)
        alt_max = config['dataset'].getfloat('40000', None)

    else:
        raise Exception('configuration file does not have a [experiment] section')

    return the_config
# parse_config_file('./config/template.ini')
