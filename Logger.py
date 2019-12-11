import os
import numpy as np
# import errno
# import torchvision.utils as vutils
# from tensorboardX import SummaryWriter
# from IPython import display
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import torch


'''
    TensorBoard Data will be stored in './runs' path
'''
def create_subdirs(root, subdirs):

    for d in subdirs:
        this_dir = os.path.join(root, d)
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)


class Logger:

    def __init__(self, model_name, dir, model, epoch=0, post_process=False):
        self.model_name = model_name
        self.dir = dir
        self.is_post_process = post_process

        if post_process is False:
            create_subdirs(dir, ('checkpoint', 'plots'))
        else:
            # create_subdirs(dir, ('results'))
            create_subdirs(os.path.join(dir,'results'),('plots','tables'))

        csvname = os.path.join(dir, model_name +'-e' + str(epoch)+ '.csv')
        self.csv = open(csvname, 'w')
        if self.csv is None:
            raise Exception('unable to open csv file ' + csvname + ' for writing')
        self.the_model = model

    def qdisplay_header(self):
        hd = '\n{:>10s}, {:3s}, {:15s}, {:10s}, {:10s}, {:10s}, {:10s}'.format(
            't', 'epoch', 'batch/nbatches', 'd_loss', 'g_loss', 'D_LOSS', 'G_LOSS')
        print(hd)
        self.csv.write(hd)

    def qdisplay_status2(self, t, epoch, nbatch, num_batches,d_error, g_error, g_loss_acum, d_loss_acum):
        print('\n{:10.0f}, {:03d}, {:7d}/{:7d}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}'.format(
            t, epoch, nbatch, num_batches, d_error, g_error, d_loss_acum, g_loss_acum)
        )

    def qdisplay_status(self, t, epoch, g_loss_acum, d_loss_acum):
        st = '\n{:10.0f}, {:03d}, {:10.4f}, {:10.4f}'.format(
            t, epoch, d_loss_acum, g_loss_acum)
        print(st)
        self.csv.write(st)

    def save_models(self, epoch):
        name='{}-e{:04d}.pth'.format(self.model_name, epoch)
        full_filename=os.path.join(self.dir, 'checkpoint',name)
        self.the_model.save(full_filename, epoch)

    def _plot1D(self,ax, data,batches, color='blue',width=0.5):
        xpoints = np.arange(len(data[0]))
        for i in range(batches):
            ypoints = data[i, :, 0].detach().cpu().numpy()
            ax.plot(xpoints, ypoints, color=color, linewidth=width)

    def _plot2D(self,ax, data,batches, color='blue',width=0.5):
        for i in range(batches):
            ypoints = data[i, :, 0].detach().cpu().numpy()
            xpoints = data[i, :, 1].detach().cpu().numpy()
            ax.plot(xpoints, ypoints, color=color, linewidth=width)

    def _dual_plot(self, plt1, plt2, data, batches, color='blue', width=0.5, legend=None):
        for i in range(batches):
            ypoints = data[i, :, 0].detach().cpu().numpy()
            xpoints = data[i, :, 1].detach().cpu().numpy()
            zpoints= data[i, :, 2].detach().cpu().numpy()
            hlines, = plt1.plot(xpoints, ypoints, color=color, linewidth=width)
            vlines,= plt2.plot(zpoints, color=color, linewidth=width)
            if legend is not None:
                hlines.set_label(legend)
                vlines.set_label(legend)
                plt1.legend()
                plt2.legend()

    def plot(self, epoch, gen_data, real_data, nsamples=10, title=None, force2D=False):
        fig = plt.figure()
        ax = plt.axes()
        batches = min(nsamples, len(gen_data))
        nfeatures=gen_data.size(2)
        if force2D is True:
            nfeatures = min(nfeatures,2)

        custom_lines = [Line2D([0], [0], color='grey', lw=4),
                        Line2D([0], [0], color='blue', lw=4)]

        if nfeatures == 1:
            self._plot1D(ax, data=real_data, batches=batches, color='grey', width=0.2)
            self._plot1D(ax, data=gen_data, batches=batches, color='blue', width=0.5)

        elif nfeatures == 2:
            self._plot2D(ax, data=real_data, batches=batches, color='grey', width=0.1)
            self._plot2D(ax, data=gen_data, batches=batches, color='blue', width=0.1)
            plt.legend(custom_lines, ['Real', 'Fake'])
            plt.xlabel('normalized longitude')
            plt.ylabel('normalized latitude')
            # plt.axis('equal')

            # if title is not None:
            #     plt.title(title)

        elif nfeatures == 3:
            newsize=fig.get_size_inches()*[2,1]
            fig.set_size_inches(newsize[0], newsize[1])
            trj_plot = plt.subplot(121)
            alt_plot = plt.subplot(122)
            # trj_plot.axis('equal')
            # alt_plot.axis('equal')
            self._dual_plot(trj_plot, alt_plot, data=real_data, batches=batches, color='grey', width=0.1,legend='real')
            self._dual_plot(trj_plot, alt_plot, data=gen_data, batches=batches, color='blue', width=0.1, legend='fake')
            fig.text(0.3, 0.04, 'normalized longitude', ha='center', va='center')
            fig.text(0.06, 0.5, 'normalized latitude', ha='center', va='center', rotation='vertical')
            fig.text(0.72, 0.04, 'trajectory sample', ha='center', va='center')
            fig.text(0.5, 0.5, 'normalized altitude', ha='center', va='center', rotation='vertical')
            fig.text(0.3, 0.9, '2D trajectory', ha='center', va='center')
            fig.text(0.72, 0.9, 'altitude profile', ha='center', va='center')

            trj_plot.legend(custom_lines, ['Real', 'Fake'])
            alt_plot.legend(custom_lines, ['Real', 'Fake'])

        if title is not None:
            fig.text(0.5, 0.95, title, weight='bold', ha='center', va='center')

        name='{}-e{:04d}.png'.format(self.model_name, epoch)
        if self.is_post_process is True:
            full_filename=os.path.join(self.dir,'results', 'plots',name)
        else:
            full_filename = os.path.join(self.dir, 'plots', name)
        fig.savefig(full_filename, bbox_inches='tight')
        plt.close()

    def on_epoch(self, t, epoch, g_loss, d_loss, gen_data, real_data):
        if epoch==0:
            self.qdisplay_header()
        self.qdisplay_status(t, epoch, g_loss, d_loss)
        self.save_models(epoch)
        self.plot(epoch, gen_data, real_data)
        self.csv.flush()

    def close(self):
        self.csv.close()