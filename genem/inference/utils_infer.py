
import functools, torch, cv2, mrcfile, os
from genem.models.networks import UnetGenerator, MaskInformedSampleF
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from genem.inference.config_infer import clean_mrc_dict, weight_map_dict
import matplotlib.patches as patches
from genem.utils import mkdirs

import logging

logger = logging.getLogger(__name__)

def get_model(load_path):
    norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=False, track_running_stats=False)
    net = UnetGenerator(1, 1, 10, 64, norm_layer=norm_layer, use_dropout=False)
    net.load_state_dict(torch.load(load_path))  
    return net

def get_adaptor(load_path, device):
    empty_feature = [
        torch.zeros((1,64, 512, 512)).float(),
        torch.zeros((1,128, 256, 256)).float(),
        torch.zeros((1,256, 128, 128)).float(),
        torch.zeros((1,512, 64, 64)).float(),
        torch.zeros((1,512, 32, 32)).float(),
    ]
    net = MaskInformedSampleF(use_mlp=True, init_type='normal', gpu_ids=[device], init_gain=0.02, nc=256)
    net(empty_feature, only_init=True)
    net.load_state_dict(torch.load(load_path))
    return net

def get_template_clean_mrc(dataset):
        return clean_mrc_dict[dataset]
    
def read_image(path):
    if path.endswith('.mrc'):
        with mrcfile.open(path, permissive=True) as mrc:
            return mrc.data.copy()
    else:
        img = Image.open(path)
        img = img.convert('L')
        return np.array(img)
    
def write_image(path, img, vmin=None, vmax=None):
    mkdirs(path)
    if path.endswith('.mrc'):
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(img)
    else:
        if vmin is not None and vmax is not None:
            img = np.clip(img, vmin, vmax)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(path, img)

def paste_particle_at_center(micrograph, particle):
    nx, ny = micrograph.shape[-2:]
    px, py = particle.shape
    micrograph[..., nx//2-px//2:nx//2+(px-px//2), ny//2-py//2:ny//2+(py-py//2)] += particle      
    return micrograph

def crop_particle_from_center(micrograph, particle_shape):
    nx, ny = micrograph.shape[-2:]
    px, py = particle_shape
    particle = micrograph[..., nx//2-px//2:nx//2+(px-px//2), ny//2-py//2:ny//2+(py-py//2)]
    return particle

def get_random_weight_map(dataset):
    weight_maps = os.listdir(weight_map_dict[dataset])
    weight_map = np.random.choice(weight_maps)
    return os.path.join(weight_map_dict[dataset], weight_map)

def convert_particle_to_frame(particle, frame_size=(100, 100)):
    """
    Converts a particle representation to a frame (image), removing white borders.
    """
    fig = Figure(figsize=(frame_size[0], frame_size[1]), dpi=1)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])  # Add an axis that covers the whole figure

    # Assuming particle data is suitable for imshow. Modify accordingly.
    ax.imshow(particle, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax.set_axis_off()  # Turn off axis

    canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

def gen_video(images, video_path, fps=30, normalize=True):
    mkdirs(video_path)
    video_size = (images.shape[1], images.shape[2])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, video_size)
    if normalize:
        images = (images - images.min()) / (images.max() - images.min())

    for image in tqdm(images, desc="Saving video"):
        frame = convert_particle_to_frame(image, video_size)
        video.write(frame)

    video.release()
    logger.info(f"Video saved to {video_path}")

def get_mask(micrograph, threshold=0.1):
    mic_norm = (micrograph - micrograph.min()) / (micrograph.max() - micrograph.min())
    mic_norm = mic_norm > threshold
    mic_norm = (mic_norm * 255).astype(np.uint8)
    return mic_norm


def visualize_sample_point(image, pos_grids, neg_grids, save_path):
    size = image.shape[0]
    def grid_to_pixel(grid, size):
        pixel = ((grid + 1) / 2) * (size - 1) 
        pixel = pixel.long()
        return pixel[..., [1, 0]]
    pos_pixels = grid_to_pixel(pos_grids, size)[0, 0]  
    neg_pixels = grid_to_pixel(neg_grids, size)[0, 0]
    box_size = 20
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')  
    for pos_pixel in pos_pixels:
        rect = patches.Rectangle((pos_pixel[1]-box_size/2, pos_pixel[0]-box_size/2), box_size, box_size, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    for neg_pixel in neg_pixels:
        rect = patches.Rectangle((neg_pixel[1]-box_size/2, neg_pixel[0]-box_size/2), box_size, box_size, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    
    mkdirs(save_path)
    plt.savefig(save_path)

