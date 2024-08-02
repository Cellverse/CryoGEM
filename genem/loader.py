import mrcfile, json, os, logging
from tqdm import tqdm 
import numpy as np

logger = logging.getLogger(__name__)

def load_particle_information_matched_with_micrographs(infos_path, mics_path, crop_size, maxH, maxW, mics):
    """
    Load particle information matched with micrographs.
    Args:
        infos_path (str): Path to the folder containing particle information.
        mics_path (str) : All mics should be matched with particle information.

        particle_info (list): List of particle information containing sub-list for each micrograph.
        each sub-list contains dict:
            'particle_idx': exact idx in generated particle stack
            'center_dim0' : center of particle in dim0
            'center_dim1' : center of particle in dim1
            'rotation'    : rotation of particle [3, 3] matrix 
        For example:
        [
            [
                {'particle_center': 100, 'center_dim1': 200, 'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                ...
            ],
            ...
        ]
    """
    all_infos = os.listdir(infos_path)
    all_infos.sort()
    matched_idx_start, matched_idx_end = [], []
    particle_center = []
    particle_rotation = []
    particle_crop_image = []

    total_num_particles = 0
    for i, mic_path in enumerate(mics_path):
        micrograph_name = os.path.basename(mic_path).split('.')[0]
        info_name = micrograph_name + '.json'
        if info_name not in all_infos:
            raise ValueError('Cannot find info file {} for micrograph {}'.format(info_name, micrograph_name))
        with open(os.path.join(infos_path, info_name), 'r') as f:
            info = json.load(f).copy()
        matched_idx_start.append(total_num_particles)
        for value in info.values():
            center_dim0 = value['center_dim0']
            center_dim1 = value['center_dim1']
            rotation = value['rotation']

            if center_dim0 < crop_size//2 or center_dim0 > maxH - crop_size//2:
                continue
            if center_dim1 < crop_size//2 or center_dim1 > maxW - crop_size//2:
                continue

            particle_center.append([center_dim0, center_dim1])
            particle_rotation.append(rotation)
            particle_crop_image.append(
                mics[i, center_dim0-crop_size//2:center_dim0+crop_size//2, center_dim1-crop_size//2:center_dim1+crop_size//2]
            )
            total_num_particles += 1
        matched_idx_end.append(total_num_particles)

        
    particle_center = np.stack(particle_center, axis=0)          # [N, 2]    
    particle_rotation = np.stack(particle_rotation, axis=0)      # [N, 3, 3]
    particle_crop_image = np.stack(particle_crop_image, axis=0)  # [N, crop_size, crop_size]

    matched_particle_info = []
    for i in range(len(mics_path)):
        temp = {
            'particle_center'    : particle_center[matched_idx_start[i]:matched_idx_end[i], :],
            'particle_rotation'  : particle_rotation[matched_idx_start[i]:matched_idx_end[i], :, :],
            'particle_crop_image': particle_crop_image[matched_idx_start[i]:matched_idx_end[i], :, :],
        }
        matched_particle_info.append(temp)

    return matched_particle_info
            
def load_micrographs_from_list(mics_path, max_num=None):
    mics_data = []
    if max_num is not None and len(mics_path) > max_num:
        mics_path = mics_path[:max_num]
    for mic_path in tqdm(mics_path, desc='Loading micrographs'):
        mics_data.append(mrcfile.open(mic_path, permissive=True).data.copy())
    mics_data = np.stack(mics_data, axis=0)
    return mics_data, mics_path

def load_micrographs_from_directory(mics_dir, max_num=None):
    mics_path = [os.path.join(mics_dir, f) for f in os.listdir(mics_dir) if f.endswith('.mrc')]
    mics_path.sort()
    return load_micrographs_from_list(mics_path, max_num=max_num)
