"""
Visualize latent space and generate volumes
"""

import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cryodrgn #import utils, config, fft
from cryodrgn.models import HetOnlyVAE
from tqdm import tqdm
import mrcfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from cryogem.lie_tools import AffineGenerator, quaternion2Rmatrix, RotateProject, AffineUniformGenerator
from cryogem import fft

logger = logging.getLogger(__name__)

class PairGenerator(Dataset):
    def __init__(self, n_particles, all_rots, all_z):
        assert all_rots.shape[0] == n_particles
        assert all_z.shape[0] == n_particles
        self.n_particles = n_particles
        self.rots = all_rots
        self.z_values = all_z
        
    def __len__(self):
        return self.n_particles
    
    def __getitem__(self, idx):
        return self.rots[idx], self.z_values[idx]
    
class ParticleGenerator:
    """Helper class to call analysis.gen_volumes"""

    def __init__(self, weights, cfg, device=None, particle_args={}):
        self.weights = weights
        
        cfg = cryodrgn.config.load(cfg)
        self.cfg = cfg
        
        self.particle_args=particle_args
                
        if device is not None:
            device = torch.device(device)
        else:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            logger.info("Use cuda {}".format(use_cuda))
            if not use_cuda:
                logger.warning("WARNING: No GPUs detected")
        
        self.D = self.cfg["lattice_args"]["D"]  # image size + 1
        self.device = device
        zdim = self.cfg["model_args"]["zdim"]
        norm = self.cfg["dataset_args"]["norm"]
        
        if self.particle_args['D'] is not None:
            assert self.particle_args['D'] % 2 == 0, "Boxsize must be even"
            assert self.particle_args['D'] <= self.D - 1, "Must be smaller than original box size"
        
        self.model, self.lattice = HetOnlyVAE.load(self.cfg, self.weights, device=device)
        self.model.eval()
    
    @torch.no_grad()
    def gen_particles(self, rots, z_values):
        B = rots.shape[0]
        D = self.D
        mask = self.lattice.get_circular_mask(self.D // 2)
        
        # logger.info(f'self.lattice.coords type', type(self.lattice.coords), self.lattice.coords.dtype)
        # logger.info(f'self.lattice.extent type', type(self.lattice.extent))
        # logger.info(f'rots type', type(rots), rots.dtype)
        # logger.info(f'z_values type', type(z_values), z_values.dtype)
        
        yhat_ht = self.model(((self.lattice.coords[mask] / self.lattice.extent / 2) @ rots), z_values).view(B, -1)
        yhat_ht_zero = torch.zeros(B,D,D).view(B,-1).to(self.device)
        yhat_ht_zero[:,mask] = yhat_ht
        yhat_ht_final = yhat_ht_zero.view(B,D,D)
        # yhat_real_ft_ift = fft.torch_ifft2_center(yhat_ht_final[:, :-1, :-1])
        
        yhat_real = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(yhat_ht_final[:, :-1, :-1], dim=(-1, -2))), dim=(-1, -2)).view(B, D-1, D-1)
        yhat_real /= yhat_real.shape[-1] * yhat_real.shape[-2]
        yhat_real = yhat_real.real - yhat_real.imag
        return yhat_real.detach().cpu().float().numpy()

def generate_hetero_data(workdir, epoch, n_particles, rotations, downsample_boxsize=None, device=None):
    
    E = epoch
    zfile = f"{workdir}/z.{E}.pkl"
    weights = f"{workdir}/weights.{E}.pkl"
    cfg = (
        f"{workdir}/config.yaml"
        if os.path.exists(f"{workdir}/config.yaml")
        else f"{workdir}/config.pkl"
    )
    
    if E == -1:
        zfile = f"{workdir}/z.pkl"
        weights = f"{workdir}/weights.pkl"

    z = cryodrgn.utils.load_pkl(zfile)
    zdim = z.shape[1]
    
    z_values_idx = np.random.randint(0,z.shape[0], n_particles)
    z_values = z[z_values_idx,:]
        
    particle_args={"n_particles": n_particles, "D": downsample_boxsize}
    
    pg = ParticleGenerator(weights, cfg, device, particle_args)
    
    pair_dataloader = DataLoader(PairGenerator(n_particles, all_rots=rotations, all_z=z_values), batch_size=8)
    
    particle_stack = []
    
    for rots, z_vals in tqdm(pair_dataloader):
        
        rots = rots.to(device)
        z_vals = z_vals.to(device)
        
        particle_stack.append(pg.gen_particles(rots,z_vals))
    
    particle_stack = np.concatenate(particle_stack, axis=0)
    return particle_stack, z_values

def generate_hetero_data_given_conform_pose(workdir, epoch, n_particles, z_values, rotations, downsample_boxsize=None, device=None):
    E = epoch
    
    weights = f"{workdir}/weights.{E}.pkl"
    cfg = (
        f"{workdir}/config.yaml"
        if os.path.exists(f"{workdir}/config.yaml")
        else f"{workdir}/config.pkl"
    )
    
    if E == -1:
        weights = f"{workdir}/weights.pkl"
    
    particle_args={"n_particles": n_particles, "D": downsample_boxsize}
    pg = ParticleGenerator(weights, cfg, device, particle_args)
    pair_dataloader = DataLoader(PairGenerator(n_particles, all_rots=rotations, all_z=z_values), batch_size=1)
    
    particle_stack = []
    
    # for rots, z_vals in tqdm(pair_dataloader):
    for rots, z_vals in pair_dataloader:
        rots = rots.to(device)
        z_vals = z_vals.to(device)
        
        particle_stack.append(pg.gen_particles(rots,z_vals))
    
    particle_stack = np.concatenate(particle_stack, axis=0)
    return particle_stack

def downsample_volume(old, D):

    assert old.shape[0] > D, "Volume size must be larger than downsample size, now {} vs {}".format(old.shape[0], D)
    oldD = old.shape[0]
    start = int(oldD / 2 - D / 2)
    stop = start + D
    
    oldft = fft.htn_center(old)
    logger.info(oldft.shape)
    newft = oldft[start:stop, start:stop, start:stop]
    logger.info(newft.shape)
    new = fft.ihtn_center(newft).astype(np.float32)
    logger.info(new.shape)
    
    return new

def downsample_image(old, D):
    """ Downsample 2d array using fourier transform """
    assert old.shape[-1] >= D, "Image size must be no less than the expected downsample size, now {} vs {}".format(old.shape[-1], D)
    oldD = old.shape[-1]
    start = int(oldD / 2 - D / 2)
    stop = start + D
    logger.info(old.shape)
    output_imgs = []
    for image in tqdm(old, desc='Downsampling'):
        oldft = fft.htn_center(image[0])
        newft = oldft[start:stop, start:stop]
        new = fft.ihtn_center(newft).astype(np.float32)
        output_imgs.append(new)
    output_imgs = np.stack(output_imgs, axis=0)
    output_imgs = output_imgs[:, None, :, :]
    logger.info(output_imgs.shape)
    return output_imgs

def generate_rotations(
    n_projections,rot_quat=None
):
    
    if rot_quat is None:
        affine_generator  = AffineGenerator(n_projections)
        return affine_generator.rotations.detach().cpu().numpy()
    
    else:
        
        quat = np.array(rot_quat)
        # quat = random_quaternion(1)
        # print(quat)
        rot = np.transpose(quaternion2Rmatrix(quat[np.newaxis,...]), axes=(2, 0, 1))
        # rot = np.transpose(quaternion2Rmatrix(quat), axes=(2, 0, 1))
        return np.tile(rot,(n_projections, 1,1)).astype(np.float32)
    
def generate_data(
    vol,
    n_projections,
    device,
    batch_size=1,
    affine_data_loader=None,
    symmetry="C1"
):
    sidelen = vol.shape[0]
    use_cuda = torch.cuda.is_available()
    vol_tensor = torch.tensor(vol).view(1, 1, sidelen, sidelen, sidelen).to(device)

    if affine_data_loader is None:
        affine_generator  = AffineGenerator(n_projections, symmetry=symmetry)
        affine_data_loader = DataLoader(affine_generator, batch_size=batch_size)
    
    rotate_project = RotateProject(sidelen).to(device)
    # rotate_project = nn.DataParallel(rotate_project)
    
    particle_stack = []
    
    for rotations in tqdm(affine_data_loader):
        rotations = rotations.to(device)
        batch = rotations.shape[0]
        clean_projected_images = rotate_project(vol_tensor.expand(batch, -1, -1, -1, -1), rotations).detach().cpu().numpy()

        particle_stack.append(clean_projected_images)
        
    particle_stack = np.stack(particle_stack, axis=0)
    return particle_stack, affine_generator.rotations.detach().cpu().numpy()

def generate_data_cryoet(
    vol,
    n_projections,
    batch_size=1,
    affine_data_loader=None
):
    sidelen = vol.shape[0]
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:1' if use_cuda else 'cpu')
    vol_tensor = torch.tensor(np.transpose(vol, axes=(2, 1, 0))).view(1, 1, sidelen, sidelen, sidelen).to(device)

    if affine_data_loader is None:
        affine_generator = AffineUniformGenerator(n_projections)
        affine_data_loader = DataLoader(affine_generator, batch_size=batch_size)
    
    rotate_project = RotateProject(sidelen).to(device)
    # rotate_project = nn.DataParallel(rotate_project)
    
    particle_stack = []
    
    for rotations in tqdm(affine_data_loader):
        rotations = rotations.to(device)
        batch = rotations.shape[0]
        clean_projected_images = rotate_project(vol_tensor.expand(batch, -1, -1, -1, -1), rotations).detach().cpu().numpy()
        # plt.imsave('cryoet.png', clean_projected_images[0], cmap='gray')
        # exit()

        particle_stack.append(clean_projected_images)
        
    particle_stack = np.stack(particle_stack, axis=0)
    return particle_stack, affine_data_loader

def padding_images(images, pad_size):
    # images: [B, 1, N, N]
    # pad_size: int
    results = []
    for image in images:
        image = np.pad(image[0], ((pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=0)
        results.append(image)

    results = np.stack(results, axis=0)
    results = results[:, np.newaxis, :, :]

    return results

def paste_image(big_image, small_image, x, y):
    """ Paste a small image onto a big image at a specified location.
    
    Args:
        big_image (np.ndarray): Big image.
        small_image (np.ndarray): Small image.
        x (int): X coordinate of the top-left corner of the big image.
        y (int): Y coordinate of the top-left corner of the big image.
    """
    mH, mW = big_image.shape[:2]
    pH, pW = small_image.shape[:2]

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + pH, mH)
    y2 = min(y + pW, mW)

    px1 = x1 - x
    py1 = y1 - y
    px2 = x2 - x
    py2 = y2 - y

    big_image[x1:x2, y1:y2] += small_image[px1:px2, py1:py2]

def generate_particles_homo(
    input_map,
    n_particles,
    particle_size,
    device,
    symmetry="C1"
):
    vol = mrcfile.open(input_map).data.astype(np.float32)
    projections, rotations = generate_data(vol, n_particles, device, symmetry=symmetry)
    projections = downsample_image(projections, particle_size)

    return projections[:, 0, :, :], rotations
    
def generate_particles_hetero(
    n_particles,
    particle_size,
    device,
    drgn_dir,
    drgn_epoch,
    same_rot,
):
    
    rot_quat =  [0.6590068, -0.28052469, 0.41374563, 0.56198798] if same_rot else None
    rotations = generate_rotations(n_particles, rot_quat)
    projections, z_values = generate_hetero_data(drgn_dir, drgn_epoch, n_particles, rotations, downsample_boxsize=particle_size, device=device)
    
    return projections, rotations, z_values
    
    
def image2mask(image, threshold=0.9, inverse_colour=True):
    image = np.array(image, dtype=np.float32)
    if inverse_colour:
        image = -image
    image = (image - image.min()) / (image.max() - image.min())
    
    mask = image < threshold 
    return mask