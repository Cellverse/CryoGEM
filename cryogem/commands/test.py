import os, torch, starfile, time, pickle, logging , argparse
import numpy as np
import pandas as pd
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from cryogem.options import process_opt, base_add_args
from cryogem.datasets import create_dataset
from cryogem.models import create_model
from cryogem.utils import make_dirs, save_as_mrc, save_as_png
from cryogem import models
from cryogem import datasets
from cryogem.config import _select_model, _select_dataset

logger = logging.getLogger(__name__)

def extra_add_args(parser):
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    # Dropout and Batchnorm has different behavioir during training and test.
    parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
    parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
    parser.add_argument('--generate_shift', default=False, action='store_true', help='generate shift images')
    parser.add_argument('--pixel_shift_max', default=5, type=int, help='max pixel shift for shift images')
    parser.add_argument('--save_dir', type=str, help='path to save generated micrographs')
    parser.add_argument('--store_real_A_trans', default=False, action='store_true', help="whether to save image translation if exists")
    parser.add_argument('--store_realA_ps', action='store_true', help='if specified, then store particle stack of real micrographs')
    parser.add_argument('--store_fakeB_ps', action='store_true', help='if specified, then store particle stack of fake micrographs')
    parser.add_argument('--store_realA_rot', action='store_true', help='if specified, then store rotation matrix of real micrographs')
    parser.add_argument('--starfile', type=str, help='path to starfile')
    parser.add_argument('--starfile_upsample', type=int, default=1, help='upsample factor for starfile')
    return parser
      
def add_args(parser):
    parser = base_add_args(parser)
    parser = extra_add_args(parser)
    parser = models.get_option_setter(_select_model)(parser, is_train=False)
    parser = datasets.get_option_setter(_select_dataset)(parser, is_train=False)
    return parser

def main(args):
    opt = process_opt(args)        # get testing options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    
    assert len(opt.gpu_ids) == 1, 'only support single GPU'
    opt.display_id = -1   # no visdom display;

    model.setup(opt)      
    assert opt.batch_size == 1, 'only support batch_size=1'
    N = len(dataset)
    logger.info(f'Generate {N} fakeB ...')
    
    save_dir            = os.path.join(opt.save_dir, opt.model, opt.name)
    mrc_save_names = ['fake_B']
    png_save_names = model.visual_names
    
    particle_limit_num = 100000 # limit the number of particles to save
    particle_table = {
        'rlnCoordinateX': [],
        'rlnCoordinateY': [],
        'rlnMicrographName': [],        
        'rlnRandomSubset': [],
    }
    mrc_save_dir_dict = {}
    png_save_dir_dict = {}
    fake_B_ps, gt_rot, gt_trans = [], [], []
    clean_real_A_ps = []

    for name in mrc_save_names:
        mrc_save_dir_dict[name] = os.path.join(save_dir, f'mics_mrc_{name}')
        make_dirs(mrc_save_dir_dict[name])
    for name in png_save_names:
        png_save_dir_dict[name] = os.path.join(save_dir, f'mics_png_{name}')
        make_dirs(png_save_dir_dict[name])
    
    
    start_time = time.time()
    model_inference_time, cur_iters, avg_save_time, avg_inf_time = 0, 0, 0, 0

    print_iters = len(dataset) // 10

    for i, data in enumerate(dataset):
        if i % print_iters == 0:
            logger.info(f'Generate {cur_iters}/{len(dataset)} fakeB, avg_save: {avg_save_time: .3f}s/it, avg_inf: {avg_inf_time: .3f}s/it, total : {time.time() - start_time: .3f}s')

        avg_save_time = (time.time() - start_time - model_inference_time) / (cur_iters + 1e-8)
        avg_inf_time = model_inference_time / (cur_iters + 1e-8)

        # model inference
        st_t = time.time()
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        model_inference_time += time.time() - st_t

        visuals = model.get_current_visuals()
        B, C, H, W = visuals['real_A'].shape
        cur_iters += B
        
        # save visuals
        img_name = f'gen_{i:04d}'
        
        for name in mrc_save_names:
            image = visuals[name][0, 0].cpu().numpy()
            image_save_path = os.path.join(mrc_save_dir_dict[name], f'{img_name}.mrc')
            save_as_mrc(image, image_save_path)
        
        for name in png_save_names:
            image = visuals[name][0, 0].cpu().float().numpy()
            image_save_path = os.path.join(png_save_dir_dict[name], f'{img_name}.png')
            if name == 'mask_A' or 'weight_map' in name:
                save_as_png(image, image_save_path, normalize=False)    
            else:
                save_as_png(image, image_save_path, normalize=True)    
        
        fake_B = visuals['fake_B'][0, 0].cpu().numpy()
        clean_real_A = visuals['clean_real_A'][0, 0].cpu().numpy()
        
        if 'mask_A' not in png_save_names:
            continue
        
        # save ps/rot
        rotations, boxes, centers = model.rotations[0], model.boxes[0], model.centers[0]
        pN = rotations.shape[0]
        shift_max = opt.pixel_shift_max 
        for j in range(pN):
            xi, yi, xa, ya = boxes[j]
            if opt.generate_shift:
                x_shift, y_shift = np.random.randint(-shift_max, shift_max+1, 2)
                xi, yi, xa, ya = xi + x_shift, yi + y_shift, xa + x_shift, ya + y_shift
            else:
                x_shift, y_shift = 0, 0
            if xi<0 or yi<0 or xa>H or ya>W:
                continue
            
            if len(fake_B_ps) < particle_limit_num:
                particle_table['rlnCoordinateX'].append(int(centers[j, 1])) #! no change to center
                particle_table['rlnCoordinateY'].append(int(centers[j, 0])) #! no change to center
                particle_table['rlnMicrographName'].append(f'{img_name}.mrc')
                xi, yi, xa, ya = int(xi), int(yi), int(xa), int(ya)
                fake_B_ps.append(fake_B[xi:xa, yi:ya])
                clean_real_A_ps.append(clean_real_A[xi:xa, yi:ya])
                gt_rot.append(rotations[j])
                gt_trans.append([y_shift, x_shift])
    
    if 'mask_A' not in png_save_names:
        exit()
    
    # save ps/rot 
    gt_rot = np.stack(gt_rot, axis=0).astype(np.float32)  # N,3,3
    gt_trans = np.stack(gt_trans, axis=0).astype(np.float32) # N,2
    N = len(particle_table['rlnCoordinateX'])
    RandomSubset = np.zeros(N, dtype=np.int32)
    particle_table['rlnCoordinateX'] = particle_table['rlnCoordinateX'] * opt.starfile_upsample
    particle_table['rlnCoordinateY'] = particle_table['rlnCoordinateY'] * opt.starfile_upsample
    particle_table['rlnRandomSubset'] = RandomSubset
    save_path = os.path.join(save_dir, 'gt_rots.npy')
    np.save(save_path, gt_rot)
    logger.info(f'Save gt_rot to {save_path}, shape: {gt_rot.shape}')
    H, W = fake_B_ps[0].shape
    gt_trans[:, 1] /= (float(H))
    gt_trans[:, 0] /= (float(W))
    
    save_path = os.path.join(save_dir, 'gt_trans.npy')
    np.save(save_path, gt_trans)
    logger.info(f'Save gt_trans to {save_path}, shape: {gt_trans.shape}')
    # save gt pose (rotations and translations)
    save_path = os.path.join(save_dir, 'gt_pose.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump((gt_rot, gt_trans), f)
    logger.info(f'Save gt_pose to {save_path}')

    star_path = os.path.join(save_dir, 'particles.star')
    particle_table = pd.DataFrame(particle_table)
    starfile.write(particle_table, star_path, overwrite=True)
    logger.info(f'Write particle table to {star_path}, total particles num: {len(particle_table)}')
    
    # save noisy particle stack
    fake_B_ps = np.stack(fake_B_ps)
    save_path = os.path.join(save_dir, 'fake_B_ps.mrc')
    save_as_mrc(fake_B_ps, save_path)
    logger.info(f'Save fakeB_ps to {save_path}, shape: {fake_B_ps.shape}')
    
    # save clean particle stack
    clean_real_A_ps = np.stack(clean_real_A_ps)
    save_path = os.path.join(save_dir, 'clean_real_A_ps.mrc')
    save_as_mrc(clean_real_A_ps, save_path)
    logger.info(f'Save clean_real_A_ps to {save_path}, shape: {clean_real_A_ps.shape}')

    logger.info(f'All {N} fakeB generated!')
    logger.info(f'Find them in {save_dir}')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())


