import logging, json, mrcfile
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class Micrograph:

    def __init__(self, micrograph_path, particle_info_path=None, particle_stack_path=None, mask_path=None):

        self.micrograph_path      = micrograph_path
        self.particle_info_path   = particle_info_path
        self.particle_stack_path  = particle_stack_path
        self.mask_path            = mask_path

        # load micrograph
        if self.micrograph_path.endswith('.mrc'):
            with mrcfile.open(self.micrograph_path, permissive=True) as f:
                self.mic = f.data
                self.mic_header = f.header
        else:
            self.mic = Image.open(self.micrograph_path).convert('L')
            self.mic = np.array(self.mic)
            
        self.mic_shape = self.mic.shape
        assert len(self.mic_shape) == 2, "Micrograph is not 2D"

        # load particle_info
        if self.particle_info_path is not None:
            with open(self.particle_info_path, 'r') as f:
                particle_info = json.load(f).copy()

            particle_center = []
            particle_rotation = []
            particle_box = []

            for value in particle_info.values():
                center_dim0 = value['center_dim0']
                center_dim1 = value['center_dim1']
                rotation = value['rotation']
                box = value['box']

                particle_center.append([center_dim0, center_dim1])
                particle_rotation.append(rotation)
                particle_box.append(box)

            self.particle_center = np.stack(particle_center, axis=0)          # [N, 2]
            self.particle_rotation = np.stack(particle_rotation, axis=0)      # [N, 3, 3]
            self.particle_box = np.stack(particle_box, axis=0)                # [N, 4]

        # load particle_stack
        if self.particle_stack_path is not None:
            with mrcfile.open(self.particle_stack_path, permissive=True) as f:
                self.particle_stack = f.data
                self.particle_stack_header = f.header
            self.particle_stack_shape = self.particle_stack.shape
            assert len(self.particle_stack_shape) == 3, "Particle stack is not 3D"

            # check if particle_stack and particle_info match by loading the first particle
            if self.particle_info_path is not None:
                load_p = self.particle_stack[0]
                p_box  = self.particle_box[0].astype(np.int).tolist()
                xi, yi, xa, ya = p_box
                crop_p = self.mic[xi:xa, yi:ya]
                eps = 1e-6
                error = np.abs(load_p - crop_p).sum()
                assert error < eps, "Particle stack and particle info do not match"

        # load mask
        if self.mask_path is not None:
            img = Image.open(self.mask_path).convert('L')
            self.mask = np.array(img).astype(bool)
            # import matplotlib.pyplot as plt
            # plt.imsave("temp_mask.png", self.mask)
            # exit()
            self.mask_shape = self.mask.shape
            assert self.mask_shape == self.mic_shape, "Mask and micrograph do not match"

    def get_micrograph(self):
        return self.mic

    def get_micrograph_path(self):
        return self.micrograph_path
    
    def get_mask(self):
        return self.mask

    def get_random_K_particles_info(self, K):
        """ get K random particles info
        """
        assert self.particle_info_path is not None, "Particle info not loaded"
        N = self.particle_center.shape[0]
        idx = np.random.choice(N, size=K, replace=False)
        rotations = self.particle_rotation[idx]
        boxes     = self.particle_box[idx]

        return rotations, boxes

    def get_all_particles_info(self):
        """ get all particles info
        """
        assert self.particle_info_path is not None, "Particle info not loaded"
        rotations = self.particle_rotation
        boxes     = self.particle_box
        centers   = self.particle_center

        return rotations, boxes, centers
        
def apply_weight_map_and_normalize(image, map, normalize_func):
    image -= image.min()
    image /= image.max()
    
    image = image * map
    image = normalize_func(image)
    
    return image


        