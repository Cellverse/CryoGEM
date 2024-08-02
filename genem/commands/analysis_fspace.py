import logging, argparse, os, cv2, torch
import numpy as np 
from tqdm import tqdm 

from genem.inference.config_infer import dataset_list, template_mask_ratio_dict, template_particle_num_dict
from genem.genem import GenEM
from genem.inference.utils_infer import gen_video
from genem.lie_tools import generate_y_rotations
from genem.ctf import generate_sequential_ctf_params, compute_safe_freqs, compute_ctf
from genem.inference.utils_infer import write_image, read_image

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--use_default", action="store_true", help="use default setting")
    parser.add_argument("--dataset", type=str, default="Ribosome(10028)", choices=dataset_list, help="default setting list")
    # genem
    parser.add_argument("--model_path", type=str, default="testing/checkpoints/Ribosome(10028)/200_net_G.pth", help="path to trained generator")
    parser.add_argument("--adaptor_path", type=str, default="testing/checkpoints/Ribosome(10028)/200_net_F.pth", help="path to trained adaptor")
    parser.add_argument("--apix", type=float, default=5.36, help="apix of input micrographs")
    parser.add_argument("--mode", type=str, default="homo", choices=["homo", "hetero"], help="Homogeneous protein(volume), Heterogeneous(cryoDRGN)")
    # homo protein
    parser.add_argument("--vol_path", type=str, default="testing/data/exp_abinitio_volumes/densitymap.10028.90.mrc", help="Homo protein volume path")
    parser.add_argument("--zxy", action="store_true", default=False, help="if true, rotate the input volume map.")
    # hetero protein
    parser.add_argument("--drgn_dir", type=str, default="testing/data/exp_abinitio_volumes/10345_neural_volume/drgn_result", help="path to drgn project directory")
    parser.add_argument("--drgn_epoch", type=int, default=49, help="cryodrgn checkpoint index")
    parser.add_argument("--zfile", type=str, default="testing/data/exp_abinitio_volumes/10345_neural_volume/drgn_result/analyze.49/pc2/z_values.txt", help="cryodrgn estimate valid z values")
    # params
    parser.add_argument("--save_dir", type=str, default="save_images/analysis_fspace/Ribosome(10028)")
    parser.add_argument("--device", type=str, default="cuda:0", help="device, e.g. cpu, cuda:0, ...")
    parser.add_argument("--df_min", type=float, default=-20000, help="Minimum defocus value")
    parser.add_argument("--df_max", type=float, default=20000, help="Maximum defocus value")
    parser.add_argument("--size", type=int, default=1024, help="Micrograph size, [size size]")
    parser.add_argument("--n_particles", type=int, default=100, help="Number of particles to generate")
    parser.add_argument("--particle_collapse_ratio", type=float, default=0.6, help="Particle collapse ratio")
    parser.add_argument("--apply_mask", action="store_true", help="if apply, generate particles where the mask is zero (black).")
    parser.add_argument("--mask_path", type=str, default="testing/data/manual_created_mask/eccv.png", help="mask path")
    #! feature vector selector 
    parser.add_argument("--feature_vector_object", type=str, default="phy", choices=["phy", "syn", "real"], 
        help="select feature vector object, physical_based(input of genem), synthetic(output of genem), real(real micrograph,without mask provided)")
    #! feature map selector 
    parser.add_argument("--feature_map_object", type=str, default="syn", choices=["phy", "syn", "real"], 
        help="select feature map object, physical_based(input of genem), synthetic(output of genem), real(real micrograph,without mask provided)")
    parser.add_argument("--real_image_path", type=str, default="testing/data/Ribosome(10028)/reals/0002.png", help="real micrograph path")
    
    return parser

WINDOW_NAME = "Analysis of GenEM's Latent Space"

POSITIVE_CIRCLE_COLOR = (255, 0, 0) # Blue,  1, 
NEGATIVE_CIRCLE_COLOR = (0, 0, 255) # Red,   0, 

class Samples:
    
    def __init__(self, size=512):
        
        self.pos_point = None 
        self.neg_point = None 
        self.size = size 
        self.current_select = True
        
    def normalize(self, x, y):
        return (x / self.size) * 2 - 1, (y / self.size) * 2 - 1
        
    def set_pos_point(self, x, y):
        self.pos_point = [x, y]
        
    def set_neg_point(self, x, y):
        self.neg_point = [x, y]
        
    def switch_current_select(self):
        self.current_select = not self.current_select
        
    def get_current_select(self):
        if self.current_select:
            return "Pos"
        else:
            return "Neg"
        
    def mouse_callback(self, event, x, y, flags, param):
        # Press button O/P
        if event == cv2.EVENT_LBUTTONDOWN: 
            if self.current_select:
                self.set_pos_point(x, y)
            else:
                self.set_neg_point(x, y)
            
    def check_complete(self):
        return self.pos_point is not None and self.neg_point is not None
    
    def draw_pos_point(self, display_image):
        if self.pos_point is None:
            return 
        point_coord = tuple(self.pos_point)
        cv2.circle(display_image, point_coord, 5, POSITIVE_CIRCLE_COLOR, -1)
    
    def draw_neg_point(self, display_image):
        if self.neg_point is None:
            return
        point_coord = tuple(self.neg_point)
        cv2.circle(display_image, point_coord, 5, NEGATIVE_CIRCLE_COLOR, -1)
        
    def get_pos_point(self):
        if self.pos_point is None:
            return None
        x, y = self.normalize(self.pos_point[0], self.pos_point[1])
        return torch.tensor([[[[x, y]]]])
    
    def get_neg_point(self):
        if self.neg_point is None:
            return None
        x, y = self.normalize(self.neg_point[0], self.neg_point[1])
        return torch.tensor([[[[x, y]]]])

import tkinter as tk
from tkinter import filedialog

def select_image_path():
    root = tk.Tk()
    root.withdraw()
    
    filetypes = [
        ('Image files', '*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.mrc'),
        ('PNG files', '*.png'),
        ('JPEG files', '*.jpg;*.jpeg'),
        ('GIF files', '*.gif'),
        ('BMP files', '*.bmp'),
        ('MRC files', '*.mrc'),
        ('All files', '*.*')
    ]

    file_path = filedialog.askopenfilename(title='Choose real micrograph path',
                                           filetypes=filetypes)

    if file_path == "":
        logger.error("No file selected.")

    return file_path

def main(args):
    if args.use_default:
        genem = GenEM(dataset=args.dataset, device=args.device)
        n_particles = template_particle_num_dict[args.dataset]
        particle_collapse_ratio = template_mask_ratio_dict[args.dataset]
        args.save_dir = f"save_images/analysis_fspace/{args.dataset}"
        args.real_image_path = None
    else:
        genem = GenEM(
            model_path=args.model_path,
            adaptor_path=args.adaptor_path,
            apix=args.apix,
            mode=args.mode,
            vol_path=args.vol_path,
            zxy=args.zxy,
            drgn_dir=args.drgn_dir,
            drgn_epoch=args.drgn_epoch,
            zfile=args.zfile,
            device=args.device,
        ) 
        n_particles = args.n_particles
        particle_collapse_ratio = args.particle_collapse_ratio
        
    size = args.size
    feature_vector_object = args.feature_vector_object
    feature_map_object = args.feature_map_object
    real_image_path = args.real_image_path
    save_dir = os.path.join(args.save_dir, f"vector.{feature_vector_object}.map.{feature_map_object}")
    
    
    if feature_vector_object != 'real' or feature_map_object != 'real':
        # generate template clean mrc
        rotations = genem.get_vol().gen_rotations(n_particles)
        particles = genem.get_vol()(rotations)
        outputs = genem.gen_template(
            size,
            particle_collapse_ratio,
            particles,
            save_name=os.path.join(args.save_dir, "template", "clean_mrc"),
        )
        clean_mrc = outputs["clean_mrc"]
        mask = outputs["mask"]  
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        write_image(os.path.join(save_dir, 'clean_mrc.png'), clean_mrc)
        write_image(os.path.join(save_dir, 'mask.png'), mask)
        logger.info(f"Save clean micrograph to {os.path.join(save_dir, 'clean_mrc.png')}")
        logger.info(f"Save mask to {os.path.join(save_dir, 'mask.png')}")
        
        physical_based = genem.apply_physical_based_function(
            clean_mrc,
            inverse_colour=True,
            to_numpy=True,
        )
        write_image(os.path.join(save_dir, 'physical_based.png'), physical_based)
        logger.info(f"Save physical based micrograph to {os.path.join(save_dir, 'physical_based.png')}")
        
        if feature_vector_object == 'syn' or feature_map_object == 'syn':
            synthetic = genem.add_noise(
                physical_based,
                is_physical=True,
            )["synthetic_micrograph"]
            write_image(os.path.join(save_dir, 'synthetic.png'), synthetic)
            logger.info(f"Save synthetic micrograph to {os.path.join(save_dir, 'synthetic.png')}")
        
    
    if feature_vector_object == 'real' or feature_map_object == 'real':
        if real_image_path is None or not os.path.exists(real_image_path):
            logger.warning("Real image path does not exist, please select the real image path.")
            real_image_path = select_image_path()
        real = read_image(real_image_path)
        write_image(os.path.join(save_dir, 'real.png'), real)
        logger.info(f"Save real micrograph to {os.path.join(save_dir, 'real.png')}")
    
    if feature_vector_object == 'phy':
        feature_vector_image = physical_based
    elif feature_vector_object == 'syn':
        feature_vector_image = synthetic
    else:
        feature_vector_image = real
    if feature_map_object == 'phy':
        feature_map_image = physical_based
    elif feature_map_object == 'syn':
        feature_map_image = synthetic
    else:
        feature_map_image = real
    
    manual_samples = Samples()
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, manual_samples.mouse_callback) 
    
    complete_select = False
    while True:
        display_image = cv2.resize(feature_vector_image, (512, 512))
        display_image = ((display_image - display_image.min()) / (display_image.max() - display_image.min()) * 255).astype(np.uint8)
        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        helper_info = f"Switch: Press SPACE | Current: {manual_samples.get_current_select()} | Complete: Press ENTER"
        text_size, _ = cv2.getTextSize(helper_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x, text_y = (display_image.shape[1] - text_size[0]) // 2, 30
        cv2.putText(display_image, helper_info, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        manual_samples.draw_pos_point(display_image) 
        manual_samples.draw_neg_point(display_image)
        
        cv2.imshow(WINDOW_NAME, display_image)
        
        if complete_select and manual_samples.check_complete():
            # save current window
            cv2.imwrite(os.path.join(save_dir, "selected_samples.png"), display_image)
            logger.info(f"Save selected samples to {os.path.join(save_dir, 'selected_samples.png')}")
            break
            
        pos_grids = manual_samples.get_pos_point()
        neg_grids = manual_samples.get_neg_point()
        
        if pos_grids is not None and feature_vector_object != 'real':
            sample = torch.nn.functional.grid_sample(mask_tensor, pos_grids, align_corners=False)[0, 0, 0, 0]
            if sample < 0.95: logger.warning(f"Positive sample at {pos_grids}, however, it is negative in mask.")
        if neg_grids is not None and feature_vector_object != 'real':
            sample = torch.nn.functional.grid_sample(mask_tensor, neg_grids, align_corners=False)[0, 0, 0, 0]
            if sample > 0.05: logger.warning(f"Negative sample at {neg_grids}, however, it is positive in mask.")
        
        key = cv2.waitKey(1)
        
        if key == ord("\r"):
            complete_select = True
        elif key == 32:
            manual_samples.switch_current_select()
            
    device = args.device 
    pos_grids = manual_samples.get_pos_point().to(device)
    neg_grids = manual_samples.get_neg_point().to(device)
    
    logger.info(f"Positive: {pos_grids}, Negative: {neg_grids}")
    
    vector_features = genem.get_feature(feature_vector_image)
    map_features = genem.get_feature(feature_map_image)
    
    feat_vec_pool_pos, feat_vec_pool_neg, _, _ = genem.adaptor(vector_features, num_patches=1, pos_grids=pos_grids, neg_grids=neg_grids, l2_norm=True, use_mlp=True)
    feat_map_pool = genem.adaptor.custom_forward(map_features, l2_norm=True, use_mlp=True)
    
    feature_levels = len(feat_map_pool)
    logger.info(f"Feature levels: {feature_levels}, save them locally, and put them in one figure finally.")
    
    fig, axs = plt.subplots(2, feature_levels, figsize=(12, 6))
    for i in range(feature_levels):
        feak_vec_pos = feat_vec_pool_pos[i][:, 0, :] # (1, 256)
        feak_vec_neg = feat_vec_pool_neg[i][:, 0, :] # (1, 256)
        feak_map_all = feat_map_pool[i] # (1, Hk, Wk, 256)
        similarity_pos = torch.einsum("bhwc,bc->bhw", feak_map_all, feak_vec_pos)
        similarity_neg = torch.einsum("bhwc,bc->bhw", feak_map_all, feak_vec_neg)
        size = similarity_pos.shape[-1]
        similarity_pos = similarity_pos[0].detach().cpu().numpy()
        similarity_neg = similarity_neg[0].detach().cpu().numpy()
        write_image(os.path.join(save_dir, f"pos.{size}.png"),similarity_pos)
        write_image(os.path.join(save_dir, f"neg.{size}.png"),similarity_neg)
        axs[0, i].imshow(similarity_pos, cmap="gray")
        axs[0, i].set_title(f"pos:{size}")
        axs[0, i].axis("off")
        axs[1, i].imshow(similarity_neg, cmap="gray")
        axs[1, i].set_title(f"neg:{size}")
        axs[1, i].axis("off")
    final_name = f"vector.{feature_vector_object}.map.{feature_map_object}.similarity.png"
    plt.savefig(os.path.join(save_dir, final_name))
    plt.close()
    logger.info(f'Final summary image at {os.path.join(save_dir, final_name)}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())

