import os, torch, logging
from cryogem import utils

logger = logging.getLogger(__name__)

def base_add_args(parser):
    """Define the common options that are used in both training and test."""
    # basic parameters
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default="0", help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    # model parameters
    parser.add_argument('--model', type=str, default='cryogem', help='chooses which model to use. [ cycle_gan | pix2pix | test | colorization | cryogem ]')
    parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=128, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [ basic | n_layers | pixel ]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='unet_1024', help='specify generator architecture [ resnet_9blocks | resnet_6blocks | unet_256 | unet_128 ]')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [ instance | batch | none ]')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [ normal | xavier | kaiming | orthogonal ]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', default=True, action='store_true', help='no dropout for the generator')
    parser.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
    parser.add_argument('--no_antialias_up', action='store_true', help='if specified, use [ upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv ]')
    # dataset parameters
    parser.add_argument('--dataset_mode', type=str, default='cryogem', help='chooses how datasets are loaded. [ unaligned | aligned | single | colorization | cryoem | cryogem ]')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--serial_batches_A', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--serial_batches_B', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--load_size', type=int, default=1024, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=1024, help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--display_winsize', type=int, default=1024, help='display window size for both visdom and HTML')
    # additional parameters
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
    # wandb parameters
    parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
    parser.add_argument('--wandb_project_name', type=str, default='CycleGAN-and-pix2pix', help='specify wandb project name')
    parser.add_argument('--apix', type=float, default=5.36, help='pixel size in angstrom, default 5.36 for empiar10028 d4x')
    parser.add_argument('--inferAB', default=False, action='store_true', help='infer both netG_A and netG_B')
    # cryogem model parameters
    parser.add_argument('--snr', default=0.1, type=float, help="If we apply gaussian noise, this parameter sets the signal-noise-ratio. In general, higher snr means less noise.")
    parser.add_argument('--custom_epoch', default=200, type=int, help='customized epoch number')
    return parser
    
def print_options(opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    logger.info(message)
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(expr_dir, exist_ok=True)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
            
def process_opt(opt):
    # process opt.suffix
    if opt.suffix:
        suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        opt.name = opt.name + suffix
    
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    opt.isTrain = opt.phase == 'train'   # train or test
    
    print_options(opt)
    
    return opt