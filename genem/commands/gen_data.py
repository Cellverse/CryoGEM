import mrcfile, os, json, cv2, time, argparse, logging 
import numpy as np
from tqdm import tqdm
from os import path as osp
from multiprocessing import Pool
from genem.utils_gen_data import generate_particles_homo, generate_particles_hetero, image2mask, paste_image

logger = logging.getLogger(__name__)

def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

def warnexists(out):
    if os.path.exists(out):
        logger.warning("Warning: {} already exists. Overwriting.".format(out))
        
def add_args(parser):
    # homo / hetero
    parser.add_argument("--mode", type=str, required=True, choices=["homo", "hetero"], help="use cryodrgn if using hetero")
    parser.add_argument("--debug", action="store_true", default=False, help="if opened, generate micrograph unparallelly.")
    # homo params
    parser.add_argument("--input_map", type=str, help="Input map file path.")
    parser.add_argument("--symmetry", type=str, default='C1', help="Symmetry of volume. [C1, D7...], used to limit generated pose space")
    # hetero params
    parser.add_argument("--drgn_dir", type=str, help="cryodrgn result directory")
    parser.add_argument("--drgn_epoch", type=int, help="checkpoint index, e.g. 49")
    parser.add_argument("--same_rot", action="store_true", default=False, help="if applied, we generate particles with same orientation but differnt heterogenelity.")
    # i/o params
    parser.add_argument("--device", type=str, default="cuda:0", help="pytorch device, e.g. cpu, cuda, cuda:0...")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save folder.")
    parser.add_argument("--n_threads", type=int, default=10, help="number of threads paralleled to generate micrographs.")
    # micrograph params
    parser.add_argument("--n_micrographs", type=int, default=10, help="Number of micrographs to generate.")    
    parser.add_argument("--micrograph_size", type=int, default=1024, help="Micrograph size. [s s]")
    #! (NOT USED) sample particles within uniform distribution, [min_particles_per_micrograph, max_particles_per_micrograph] 
    # parser.add_argument("--min_particles_per_micrograph", type=int, default=90, help="Minimum number of particles per micrograph.")
    # parser.add_argument("--max_particles_per_micrograph", type=int, default=130, help="Maximum number of particles per micrograph.")
    # sample particles within normal distribution, (mu-2sigma, mu+2sigma)
    parser.add_argument("--particles_mu", type=float, default=100, help="Mean number of particles per micrograph.")
    parser.add_argument("--particles_sigma", type=float, default=14, help="Standard deviation of number of particles per micrograph.")
    # particle params
    parser.add_argument("--n_particles", type=int, default=10000, help="Number of particles to generate.")
    parser.add_argument("--batch_size", type=int, default=1000, help="Particle batch size used for generating micrographs.")
    parser.add_argument("--particle_size", type=int, required=True, help="Required particle size. [D D]")
    parser.add_argument("--particle_collapse_ratio", type=float, default=0.5, help="Particle collapse ratio, used to control the overlap between particles, bigger value means more overlap.")
    parser.add_argument("--mask_threshold", type=float, default=0.9, help="Threshold to generate particle mask.")
    return parser

def main(opt):
    mkbasedir(opt.save_dir)
    warnexists(opt.save_dir)

    save_dir = opt.save_dir
    n_micrographs = opt.n_micrographs
    mode = opt.mode
    mask_threshold = opt.mask_threshold
    
    mics_mrc_dir           = osp.join(save_dir, 'mics_mrc')
    mics_png_dir           = osp.join(save_dir, 'mics_png')
    mics_mask_dir          = osp.join(save_dir, 'mics_mask')
    particles_mask_dir     = osp.join(save_dir, 'particles_mask')
    mics_particle_info_dir = osp.join(save_dir, 'mics_particle_info')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(mics_mrc_dir, exist_ok=True)
    os.makedirs(mics_png_dir, exist_ok=True)
    os.makedirs(mics_mask_dir, exist_ok=True)
    os.makedirs(particles_mask_dir, exist_ok=True)
    os.makedirs(mics_particle_info_dir, exist_ok=True)
    
    opt.n_particles = int(opt.particles_mu * opt.n_micrographs * 1.2)
    opt.batch_size = int(opt.particles_mu * 10)
    
    if mode == 'homo':
        resized_particles, rotations = generate_particles_homo(
            opt.input_map,
            opt.n_particles,
            opt.particle_size,
            opt.device,
            symmetry=opt.symmetry
        )
    elif mode == 'hetero':
        resized_particles, rotations, z_values = generate_particles_hetero(
            opt.n_particles,
            opt.particle_size,
            opt.device,
            opt.drgn_dir,
            opt.drgn_epoch,
            opt.same_rot
        )

    sync_mics_name = [f'sync_mic_{i:04d}' for i in range(n_micrographs)]

    if not opt.debug:
        pbar = tqdm(total=len(sync_mics_name), unit='image', desc=f'Generating synthetic micrographs (parallelly)...')

        batch_size = opt.batch_size
        epochs = opt.n_micrographs // batch_size
        if n_micrographs % batch_size != 0: epochs += 1
        for epoch in range(epochs):
            random_start = np.random.randint(0, resized_particles.shape[0] - batch_size + 1)
            start = epoch * batch_size
            end = min((epoch + 1) * batch_size, n_micrographs)
            pool = Pool(opt.n_threads)
            for name in sync_mics_name[start:end]:
                pool.apply_async(worker, args=(
                        name, 
                        resized_particles[random_start:random_start + batch_size],
                        rotations[random_start:random_start + batch_size],
                        opt.particles_mu,
                        opt.particles_sigma,
                        [opt.micrograph_size, opt.micrograph_size],
                        opt.particle_collapse_ratio,
                        random_start,
                        mics_mrc_dir,          
                        mics_png_dir,          
                        mics_mask_dir,         
                        particles_mask_dir,    
                        mics_particle_info_dir,
                        mask_threshold,
                    ), 
                    callback=lambda arg: pbar.update(1)
                )
            pool.close()
            pool.join()
            
        pbar.close()

        logger.info('All processes done.')
    
    else:
        for name in tqdm(sync_mics_name, unit='image', desc=f'Generating synthetic micrographs ...'):
            worker(
                name, 
                resized_particles,
                rotations,
                opt.particles_mu,
                opt.particles_sigma,
                [opt.micrograph_size, opt.micrograph_size],
                opt.particle_collapse_ratio,
                0,
                mics_mrc_dir,          
                mics_png_dir,          
                mics_mask_dir,         
                particles_mask_dir,    
                mics_particle_info_dir,
                mask_threshold,
            )
            
    # copy json file
    opt.resized_particles = resized_particles.shape
    opt.rotations = rotations.shape
    opt.func = None
    opt._parser = None
    with open(osp.join(save_dir, 'opt.json'), 'w') as f:
        json.dump(vars(opt), f)

    # save rotations and resized_particles
    np.save(osp.join(save_dir, 'rotations.npy'), rotations)
    np.save(osp.join(save_dir, 'particles.npy'), resized_particles)

def worker(name, 
           resized_particles,
           rotations,
           particles_mu,
           particles_sigma,
           micrograph_size,
           particle_collapse_ratio,
           particle_index_offset,
           mics_mrc_dir,          
           mics_png_dir,          
           mics_mask_dir,         
           particles_mask_dir,    
           mics_particle_info_dir,
           mask_threshold,
        ):
    """Worker for each process.

    Args:
        name (str): Sync Micrograph save name.
        opt (dict): Configuration dict. It contains:
            resized_particles (np.ndarray)        : Resized particles. [N, H', W']
            minimum_particles_per_micrograph (int): Minimum particles per micrograph.
            maximum_particles_per_micrograph (int): Maximum particles per micrograph.
            unresized_micrograph_size ([int, int]): Unresized micrograph size. [H, W]
            resize_divide_scale (float)           : Resize scale.
            particle_collapse_ratio (float)       : Particle collapse ratio.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    pN, pH, pW = resized_particles.shape
    mH, mW = micrograph_size[0], micrograph_size[1]
    particle_mask_len = int((1 - particle_collapse_ratio) * pH)
    np.random.seed(int(time.time() * 1000) % 2**32)
    # assert maximum_particles_per_micrograph <= pN, 'maximum_particles_per_micrograph should be less than or equal to pN.'
    # sampled_particles_num = np.random.randint(minimum_particles_per_micrograph, maximum_particles_per_micrograph + 1)
    # sampled_particles_idx = np.random.choice(pN, sampled_particles_num, replace=False)
    sampled_particles_num = np.random.normal(particles_mu, particles_sigma)
    sampled_particles_num = max(particles_mu - 2 * particles_sigma, min(particles_mu + 2 * particles_sigma, sampled_particles_num))
    sampled_particles_num = int(min(sampled_particles_num, pN)) # within 2sigma of (mu, sigma) distribution
    sampled_particles_start_idx = np.random.randint(0, pN-sampled_particles_num)
    sampled_particles_idx = np.arange(sampled_particles_start_idx, sampled_particles_start_idx + sampled_particles_num) # mod pN
    assert np.all(sampled_particles_idx < pN), 'sampled_particles_idx should be less than pN.'
    micrograph            = np.zeros((mH, mW), dtype=resized_particles.dtype)
    micrograph_mask       = np.zeros((mH, mW), dtype=bool)
    particle_center_mask  = np.zeros((mH, mW), dtype=bool)
    particle_infomation   = {}
    for i, idx in enumerate(sampled_particles_idx):
        coords = np.where(micrograph_mask == False)
        if len(coords[0]) == 0:
            logger.warning(f'Wish to generate {sampled_particles_num} particles, but only {i} particles can be generated.')
            break
        coords_x, coords_y = coords[0], coords[1]
        k = np.random.randint(len(coords_x))
        x, y = coords_x[k]-pH//2, coords_y[k]-pW//2
        paste_image(micrograph, resized_particles[idx], x, y)
        xi, xa = max(coords_x[k] - particle_mask_len, 0), min(coords_x[k] + particle_mask_len, mH)
        yi, ya = max(coords_y[k] - particle_mask_len, 0), min(coords_y[k] + particle_mask_len, mW)
        micrograph_mask[xi:xa, yi:ya] = True
        particle_center_mask[coords_x[k], coords_y[k]] = True
        xi, xa = max(coords_x[k] - pH//2, 0), min(coords_x[k] + (pH - pH//2), mH)
        yi, ya = max(coords_y[k] - pW//2, 0), min(coords_y[k] + (pW - pW//2), mW)
        if xa - xi != pH or ya - yi != pW:
            continue
        box = [int(xi), int(yi), int(xa), int(ya)]
        particle_infomation['particle_{}'.format(i)] = \
            {
                'particle_idx': int(idx)+particle_index_offset,
                'center_dim0': int(coords_x[k]),
                'center_dim1': int(coords_y[k]),
                'rotation': rotations[idx].tolist(),
                'box': box,
            }
    micrograph_uint8 = ((micrograph - micrograph.min()) / (micrograph.max() - micrograph.min()) * 255).astype(np.uint8)
    particle_mask = (image2mask(micrograph_uint8, threshold=mask_threshold)*255).astype(np.uint8)
    particle_center_mask = particle_center_mask.astype(np.uint8) * 255
    with mrcfile.new(osp.join(mics_mrc_dir, f'{name}.mrc'), overwrite=True) as f:
        f.set_data(micrograph)
    cv2.imwrite(osp.join(particles_mask_dir, f'{name}.png'), particle_mask)
    cv2.imwrite(osp.join(mics_png_dir, f'{name}.png'), micrograph_uint8.astype(np.uint8))
    cv2.imwrite(osp.join(mics_mask_dir, f'{name}.png'), particle_center_mask)
    with open(osp.join(mics_particle_info_dir, f'{name}.json'), 'w') as f:
        json.dump(particle_infomation, f)
    process_info = f'Processing {name} ...'
    return process_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())







