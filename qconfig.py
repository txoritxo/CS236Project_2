import configparser

def parse_config_file(filename):
    the_config = {}
    config = configparser.ConfigParser()
    config.read(filename)
    if 'experiment' in config.sections():
        the_config['id']      = config['experiment'].get('id', 'Unknown')
        the_config['dataset'] = config['experiment'].get('dataset', 'Unknown')
    else:
        raise Exception('configuration file does not have a [experiment] section')

    if 'model' in config.sections():
        the_config['latent_dimension'] = config['model'].getint('latent_dimension', 10)
        the_config['generator_type'] = config['model'].get('generator_type', 'RNN')
        the_config['generator_RNN_cell'] = config['model'].get('generator_RNN_cell', 'LSTM')
        the_config['generator_hidden_units'] = config['model'].getint('generator_hidden_units', 10)
        the_config['generator_RNN_layers'] = config['model'].getint('generator_RNN_layers', 1)

        the_config['discriminator_type'] = config['model'].get('discriminator_type', 'RNN')
        the_config['discriminator_RNN_cell'] = config['model'].get('discriminator_RNN_cell', 'LSTM')
        the_config['discriminator_hidden_units'] = config['model'].getint('discriminator_hidden_units', 10)
        the_config['discriminator_RNN_layers'] = config['model'].getint('discriminator_RNN_layers', 1)
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

# parse_config_file('./config/template.ini')