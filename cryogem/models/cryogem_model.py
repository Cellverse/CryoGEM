import numpy as np
import torch
from cryogem.models.base_model import BaseModel
from cryogem.models import networks
from cryogem.micrograph import apply_weight_map_and_normalize
from cryogem.transform import instance_normalize
from cryogem import utils, losses

import logging

logger = logging.getLogger(__name__)

class CryoGEMModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.set_defaults(no_dropout=True, no_antialias=True, no_antialias_up=True)  
        
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=10.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=utils.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='1,2,3,4,5', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=utils.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mask_sample', choices=['sample', 'reshape', 'mlp_sample', 'mask_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=utils.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        
        parser.set_defaults(pool_size=0)  # no image pooling
        parser.set_defaults(netF="mask_sample") 

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        logger.info(f'Sampler Type: {opt.netF}')

        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        
        if self.isTrain:
            self.visual_names      = ['real_A', 'noisy_real_A', 'fake_B',  'real_B', 'mask_A'] # real_A -> fake_B
            self.model_names       = ['G', 'F', 'D']
        else:
            self.visual_names      = ['real_A', 'noisy_real_A', 'fake_B', 'mask_A']
            self.model_names       = ['G'] # during test time, only load G
            
        self.visual_names += ['clean_real_A', 'weight_map']
            
        # define networks (both generator and discriminator)
        if 'G' in self.model_names:
            self.netG = networks.define_G(
                1, 
                1, 
                opt.ngf, 
                opt.netG, 
                norm=opt.norm, 
                use_dropout=not opt.no_dropout, 
                init_type=opt.init_type, 
                init_gain=opt.init_gain, 
                gpu_ids=self.gpu_ids, 
                opt=opt,
                no_antialias=opt.no_antialias,
                no_antialias_up=opt.no_antialias_up
            )
        if 'F' in self.model_names:
            self.netF = networks.define_F(
                1, 
                opt.netF, 
                norm=opt.norm, 
                use_dropout=not opt.no_dropout, 
                init_type=opt.init_type, 
                init_gain=opt.init_gain, 
                gpu_ids=self.gpu_ids, 
                opt=opt
            )
        if 'D' in self.model_names:
            self.netD = networks.define_D(
                1, 
                opt.ndf, 
                opt.netD, 
                n_layers_D=opt.n_layers_D, 
                norm=opt.norm, 
                init_type=opt.init_type, 
                init_gain=opt.init_gain, 
                gpu_ids=self.gpu_ids,
                opt=opt,
                no_antialias=opt.no_antialias
            )
            
        logger.info(self.opt.nce_layers)
        
        if self.isTrain:
            # define loss functions
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)
            if self.opt.netF != 'mask_sample':
                self.criterionNCE = losses.PatchNCELoss(opt).to(self.device)
            else:
                self.criterionNCE = losses.MaskAwaredPatchNCELoss(opt).to(self.device)
                
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.phase == 'train':
            self.compute_D_loss().backward()                   # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF in ['mlp_sample', 'mask_sample']:
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        loss_G = self.loss_G
        loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF in ['mlp_sample', 'mask_sample']:
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.clean_real_A = self.real_A
        
        self.weight_map = input['weight_map'].to(self.device)
        self.real_A = apply_weight_map_and_normalize(self.real_A, self.weight_map, instance_normalize)
        
        if self.opt.mask_dir:
            self.mask_A = input['mask_A'].to(self.device)
        
        if self.opt.phase == 'train':
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']
        else:
            self.rotations = input['rotations']
            self.boxes     = input['boxes']
            self.centers   = input['centers']

    def forward(self, encode_only=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real_A = torch.flip(self.real_A, [3])
        
        if encode_only:
            self.feats, self.physical_based_input = self.netG(
                self.real_A,  # real_A -> (ctf) -> (noise) -> encoder -> decoder -> fake_B
                encode_only=True,
                apply_ctf=True,
                apply_gaussian_noise=True,
                snr=self.opt.snr,
                apix=self.opt.apix,
            )  # G(A)
        else:
            self.fake_B, self.noisy_real_A = self.netG(
                self.real_A,  # real_A -> (ctf) -> (noise) -> encoder -> decoder -> fake_B
                encode_only=False,
                apply_ctf=True,
                apply_gaussian_noise=True,
                snr=self.opt.snr,
                apix=self.opt.apix,
            )  # G(A)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # assert fake.device == next(self.netD.parameters()).device, "fake and netD are not on the same device"

        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            object = self.noisy_real_A
            self.loss_NCE = self.calculate_NCE_loss( # make features of fake_B -> noisy_real_A
                object, 
                self.fake_B, 
                masks=self.mask_A
            )
            
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G
   

    def calculate_NCE_loss(self, src, tgt, masks=None):
        # tgt: fake_B
        feat_q, _ = self.netG( # fake_B -> encoder -> features
                tgt, 
                nce_layers=self.nce_layers,
                encode_only=True, 
                apply_ctf=False, 
                apply_gaussian_noise=False,
                apix=self.opt.apix,
        )
            
        n_layers = len(feat_q)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
            
         # src: real_A
        feat_k, _ = self.netG(  # clean_real_A/noisy_real_A -> encoder -> features
            src, 
            nce_layers=self.nce_layers,
            encode_only=True, 
            apply_ctf=False,
            apply_gaussian_noise=False,
            apix=self.opt.apix,
        )
            
        assert len(feat_k) == n_layers, 'The number of layers should be the same for the source and target'
        
        total_nce_loss = 0.0
        if self.opt.netF == 'mask_sample':
            feat_k_pool_pos, feat_k_pool_neg, pos_grids, neg_grids = self.netF(feat_k, num_patches=self.opt.num_patches, masks=masks)
            feat_q_pool_pos, feat_q_pool_neg,         _,         _ = self.netF(feat_q, num_patches=self.opt.num_patches, pos_grids=pos_grids, neg_grids=neg_grids, masks=masks)
            
            for f_q_pos, f_q_neg, f_k_pos, f_k_neg in zip(feat_q_pool_pos, feat_q_pool_neg, feat_k_pool_pos, feat_k_pool_neg):
                loss_1 = self.criterionNCE(f_q_pos, f_q_neg, f_k_pos) * self.opt.lambda_NCE
                loss_2 = self.criterionNCE(f_q_neg, f_q_pos, f_k_neg) * self.opt.lambda_NCE
                loss = loss_1.mean() + loss_2.mean()
                loss = loss / 2.0
                total_nce_loss += loss
                
        else:
            feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
            feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)          
            for f_q, f_k in zip(feat_q_pool, feat_k_pool):
                loss = self.criterionNCE(f_k, f_q) * self.opt.lambda_NCE
                total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
