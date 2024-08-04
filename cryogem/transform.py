import torch, random
import torch.nn as nn
from cryogem import filter
import logging
import numpy as np

logger = logging.getLogger(__name__)

class GaussianPyramid(nn.Module):
    def __init__(self, kernel_size=11, kernel_variance=0.01, num_octaves=4, octave_scaling=10):
        """
        Initialize a set of gaussian filters.

        Parameters
        ---------
        kernel_size: int
        kernel_variance: float
        num_octaves: int
        octave_scaling: int
        """
        super(GaussianPyramid,self).__init__()
        self.kernel_size = kernel_size
        self.variance = kernel_variance
        self.num_dec = num_octaves
        self.scaling = octave_scaling

        weighting = torch.ones([num_octaves], dtype=torch.float32)
        self.register_buffer('weighting', weighting)
        self.kernels = self.generateGaussianKernels(kernel_size, kernel_variance, num_octaves + 1, octave_scaling)

        self.gaussianPyramid = torch.nn.Conv2d(1, num_octaves + 1,
                                               kernel_size=kernel_size,
                                               padding='same', padding_mode='reflect', bias=False)
        self.gaussianPyramid.weight = torch.nn.Parameter(self.kernels)
        self.gaussianPyramid.weight.requires_grad = False

    def generateGaussianKernels(self, size, var, scales=1, scaling=2):
        """
        Generate a list of gaussian kernels

        Parameters
        ----------
        size: int
        var: float
        scales: int
        scaling: int

        Returns
        -------
        kernels: list of torch.Tensor
        """
        coords = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        xy = torch.stack(torch.meshgrid(coords, coords),dim=0)
        kernels = [torch.exp(-(xy ** 2).sum(0) / (2 * var * scaling ** i)) for i in range(scales)]
        kernels = torch.stack(kernels,dim=0)
        kernels /= kernels.sum((1, 2), keepdims=True)

        kernels = kernels[:, None, ...]
        return kernels

    def forward(self, x):
        return self.gaussianPyramid(x)

def instance_normalize(img, autocontrast=False):
    if not autocontrast:
        # img = (img - img.mean()) / (img.std() + 1e-8)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 2 - 1

        return img

    else:
        assert len(img.shape) == 3, 'img should be [C, H, W]'

        # parameters to do the preprocess #
        t_sd = 40.0/255.0
        t_mean = 150.0/255.0

        result = torch.zeros_like(img)

        for i, gray in enumerate(img):
            gray = gray / (torch.max(abs(gray)) + 1e-8)

            mean = torch.mean(gray)
            sd = torch.std(gray)

            F = t_sd / sd
            A = t_mean - F * mean
            black = -A / F
            white = (1 - A) / F

            gray = torch.clip(gray, black, white)
            gray = (gray - black) / (white - black + 1e-8) * 2 - 1 # [-1, 1]

            result[i] = gray

        return result

def custom_transform(data, sync_revs_colr=False, normalize=True, autocontrast=False, crop_size=1024, mask=None, weight_map=None, random_crop=True, left_up=False):
    """ transform for custom data
    
    Args:
        data (np.ndarray)     : data to be transformed, [H, W]
        sync_revs_colr (bool) : whether to reverse colour
        normalize (bool)      : whether to normalize the data
        autocontrast (bool)   : whether to use autocontrast
    
    """
    
    if sync_revs_colr:
        data = -data
    if normalize:
        data = instance_normalize(data, autocontrast=autocontrast) 
        # data = (data - data.min()) / (data.max() - data.min()) #! since we need to apply weight map, we need to map to [0, 1] instead of [-1, 1]
        # data = (data - data.mean()) / data.std()
    
    if not random_crop:
        if mask is not None:
            return data, mask, weight_map
        return data
        
    H, W = data.shape
    if H < crop_size or W < crop_size:
        raise ValueError('data size should be larger than crop size')
    
    # if mask is None and not left_up:
    #     cx, cy = random.randint(0, H - crop_size), random.randint(0, W - crop_size)
    # else:
    if H != crop_size:
        # points = np.where(mask[0:H - crop_size, 0:W - crop_size])
        # idx = random.randint(0, len(points[0]) - 1) 
        # cx, cy = points[0][idx], points[1][idx]
        cx, cy = random.randint(0, H - crop_size-1), random.randint(0, W - crop_size-1)
    else:
        cx, cy = 0, 0
        
    #　rotate 0 / 90 / 180 / 270
    rotate = random.randint(0, 3)
    rot_trans = lambda x: np.rot90(x, k=rotate)
    
    data = data[cx: cx + crop_size, cy: cy + crop_size]
    if not left_up:
        data = rot_trans(data).copy()
            
    if mask is not None:
        mask = mask[cx: cx + crop_size, cy: cy + crop_size]
        if not left_up:
            mask = rot_trans(mask).copy()    
        
    if weight_map is not None:
        weight_map = weight_map[cx: cx + crop_size, cy: cy + crop_size]

    if mask is not None:
        return data, mask, weight_map
    
    return data

def icebreaker_transform(data, icebreaker_params=None, return_map=False, map_len=1024):
    """ image equalization for each class of pixel-intensity-based K-mean clustering
    Args:
        data (np.ndarray) : data to be transformed, [B, C, H, W], where C=1
        lowpass_param_dict (dict) : parameters for lowpass filter with cosine edge
            contains:
                apix (float)              : pixel size in angstrom, default 5.36
        
        rolled_param_dict (dict) : parameters for rolling the image
            contains:
                x_patch_num (int)         : number of patches in x direction, default 20
                y_patch_num (int)         : number of patches in y direction, default 20
                downsample_ratio (int)    : downsample ratio before KMenas, default 40

        kmeans_param_dict (dict) : parameters for K-means clustering
            contains:
                cluster_num (int)         : number of clusters, default 8
    """
    B, C, H, W = data.shape
    assert C == 1, 'data should be [B, 1, H, W]'

    if icebreaker_params == None:
        icebreaker_params = {
            'apix': 5.36,
            'x_patch_num': 20, 
            'y_patch_num': 20, 
            'cluster_num': 8
        }

    results = []
    
    required_grad = data.requires_grad

    for i, image in enumerate(data):
        image = image[0]
        min, max = image.min(), image.max()
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) # [0, 1]
        detached_image = image.detach()
        
        #TODO: lowpass & gaussian blur
        lowpass = filter.filter_image(detached_image, cutoff_ratio=0.065*icebreaker_params['apix'])

        #TODO: rolled & resize & gaussian blur
        rolled = filter.window(lowpass, icebreaker_params['x_patch_num'], icebreaker_params['y_patch_num'])
        resized_rolled = filter.custom_resize(rolled, target_size=[256, 256])
        # resized_rolled = filter.custom_resize(rolled, downsample_ratio=icebreaker_params['downsample_ratio'])
        
        #TODO: K-means clustering &　resize
        cluster_res = filter.custom_KMeans(resized_rolled, icebreaker_params['cluster_num'])

        assert cluster_res.requires_grad == False, 'cluster_res should not require grad'

        if not return_map:
            result      = filter.custom_histogram_equalization_with_kmeans_result(lowpass, cluster_res)
            assert result.requires_grad == required_grad, 'icebreaker changed the requires_grad of image'
            result = result * (max - min) + min # [min, max] : change back to min, max
            results.append(result[None, ...]) # [1, H, W]
        else:
            segmap = filter.custom_histogram_equalization_with_kmeans_result(lowpass, cluster_res, return_map=return_map)
            segmap /= segmap.max()
            segmap = filter.custom_resize(segmap, target_size=[map_len, map_len]) # change to our target size
            results.append(segmap)

    results = torch.stack(results, dim=0)
    
    return results

def add_baseline_gaussian_nosie(img, snr=0.1):
    assert len(img.shape) == 2, 'img should be [H, W]'

    noise_std = np.sqrt(np.var(img, axis=(-2, -1), keepdims=True) / snr)
    noisy_img = np.random.normal(img, noise_std)

    return noisy_img

    

