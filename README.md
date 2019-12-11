# Generation of Aircraft Trajectories with Deep Generative Models
This repository contains the python software supporting the project "Generation of Aircraft Trajectories with Deep Generative Models" submitted as part of the CS236 course homework. Initially the project used software from the repository https://github.com/ratschlab/RGAN.git  implementing the findings of the paper_[Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633)_. As more complex models and features were needed (multi-layer RNNs, GRU units, minibatch discrimination...) it was decided to do a fresh implementation in Pytorch, which is what this repository contains.

## Defining and Training a model
Directory ./config contains a wide set of configuration files that were used for the experiments. In these configuration files there is the definition of parameters like the particular dataset file to be used, generator and discriminator model architectures and parameters, training parameters, etc. It is best to have a look at the files as they are quite self-explanatory.
Once the configuration is finished, train with the following command
`$python qqrun_experiment.py --config_file ./config/your_config_file_here.ini `

To resume training from a checkpoint file, use the following command
`$python qqrun_experiment.py --config_file ./logs/path_to_your_checkpoint_file.pth `

The training process will log inside ./logs directory the following information:
+ ** training log**, including time, epoch, generator loss, discriminator loss, GAN loss
+ **plots** of generated data vs real data for every epoch
+ **checkpoint** file of the model for every epoch

## Post-training data process
The file **qpprocess_data.py** contains several utils for the following:
+ **generate DTW** metric given a model, start and end epochs, and samples to use in the computation of the metric
+ **generate result plots** associated with a model's checkpoint, this will generate a plot of real vs generated data, the number of samples to include in the plot is used defined
+ **generate real plots** Plots random samples from a dataset. The number of samples is user defined. 

