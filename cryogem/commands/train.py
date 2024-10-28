import time, torch, logging, argparse
from tqdm import tqdm
from cryogem.options import process_opt, base_add_args
from cryogem.datasets import create_dataset
from cryogem.models import create_model
from cryogem.visualizer import Visualizer
from cryogem import models
from cryogem import datasets
from cryogem.config import _select_model, _select_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def extra_add_args(parser):
    # visdom and HTML visualization parameters
    parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
    parser.add_argument('--display_ncols', type=int, default=6, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display')
    parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
    parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
    parser.add_argument('--update_html_freq', type=int, default=100, help='frequency of saving training results to html')
    parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
    parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    # network saving and loading parameters
    parser.add_argument('--save_latest_freq', type=int, default=100000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    # training parameters
    parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=75, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--iters_per_epoch', type=int, default=100, help='number of iterations per epoch')
    return parser

def add_args(parser):
    parser = base_add_args(parser)
    parser = extra_add_args(parser)
    parser = models.get_option_setter(_select_model)(parser, is_train=True)
    parser = datasets.get_option_setter(_select_dataset)(parser, is_train=True)
    return parser

def main(args):
    opt = process_opt(args)        # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    logger.info('The number of training images = %d' % dataset_size)

    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        
        for i, data in enumerate(tqdm(dataset, desc=f"Epoch {epoch}/{opt.n_epochs}, iters: {epoch_iter}/{dataset_size}")):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            if 'cryogem' in opt.model and epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            elif epoch == opt.epoch_count and i == 0:
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                logger.info('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            
            iter_data_time = time.time()
        
        model.update_learning_rate()    # update learning rates in the end of every epoch.
            
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    model.save_networks('latest')
    model.save_networks(epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
    