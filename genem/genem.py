
from genem.inference.utils_infer import get_model, get_adaptor, read_image, write_image, paste_particle_at_center, crop_particle_from_center, get_mask
from genem.inference.config_infer import dataset_list, model_dict, adaptor_dict, apix_dict
from genem.inference.protein import Protein_volume, get_volume
from genem.micrograph import apply_weight_map_and_normalize
from genem.transform import instance_normalize
from genem.ctf import generate_random_ctf_params, compute_safe_freqs, compute_ctf, torch_fft2_center, torch_ifft2_center
from typing import Union
import numpy as np
import torch

import logging 

logger = logging.getLogger(__name__)

class GenEM:
    
    def __init__(self, 
        dataset=None,
        model_path="testing/checkpoints/Ribosome(10028)/200_net_G.pth",
        adaptor_path="testing/checkpoints/Ribosome(10028)/200_net_F.pth",
        apix=5.36,
        mode="homo",
        vol_path="testing/data/exp_abinitio_volumes/densitymap.10028.90.mrc", 
        zxy=True,
        drgn_dir="testing/data/exp_abinitio_volumes/10345_neural_volume/drgn_result",
        drgn_epoch=49,
        zfile="testing/data/exp_abinitio_volumes/10345_neural_volume/drgn_result/analyze.49/pc2/z_values.txt",
        device="cuda:0", 
    ):
        super(GenEM, self).__init__()
        if dataset is not None and dataset in dataset_list:
            self.model = get_model(model_dict[dataset]).to(device)
            self.adaptor = get_adaptor(adaptor_dict[dataset], device).to(device)
            self.apix = apix_dict[dataset]
            self.vol = get_volume(dataset, device)
        else:
            self.model = get_model(model_path).to(device)
            self.adaptor = get_adaptor(adaptor_path).to(device)
            self.apix = apix 
            self.vol = Protein_volume(
                mode=mode, device=device, 
                vol_path=vol_path, zxy=zxy,  # homo params
                drgn_dir=drgn_dir, drgn_epoch=drgn_epoch, zfile=zfile # hetero params
            )
            
        self.device = device
        
    def get_vol(self):
        return self.vol
    
    def apply_physical_based_function(
        self,
        clean_mrc,
        weight_map=None,
        particle=None,
        ctf=None,
        inverse_colour=True,
        to_numpy=False,
    ):
        device = self.device
        if weight_map is not None and isinstance(weight_map, str):
            weight_map = read_image(weight_map)
        sidelen = clean_mrc.shape[0]
        if particle is not None: clean_mrc = paste_particle_at_center(clean_mrc, particle)
        if inverse_colour: clean_mrc = -clean_mrc # inverse colour
        if weight_map is not None: clean_mrc = apply_weight_map_and_normalize(clean_mrc, weight_map, instance_normalize)
        if ctf is None:
            ctf_params = generate_random_ctf_params(1)
            freqs_mag, angles_rad = compute_safe_freqs(sidelen, self.apix)
            ctf = compute_ctf(freqs_mag, angles_rad, *ctf_params).reshape(1, sidelen, sidelen)
        ctf = torch.from_numpy(ctf).float().to(device)
        input = torch.from_numpy(clean_mrc).float().unsqueeze(0).unsqueeze(0).to(device)
        ctf_corrupted_fourier_images = ctf * torch_fft2_center(input)
        input = torch_ifft2_center(ctf_corrupted_fourier_images).real
        if to_numpy:
            input = input.detach().cpu().numpy()[0,0,...]
        return input
    
    def get_feature(
        self, 
        input:np.ndarray,
    ):
        input = torch.from_numpy(input).float().unsqueeze(0).unsqueeze(0).to(self.device)
        outputs = self.model( 
                input,
                apply_ctf=False,
                apply_gaussian_noise=False,
                snr=0.1,
                apix=self.apix,
                nce_layers=[1,2,3,4,5],
                return_features=True,
        )
        input_features, output_features, synthetic, physical_based = outputs
        return input_features
        
    def add_noise(
        self,
        clean_mrc: Union[str, np.ndarray],
        weight_map: Union[str, np.ndarray]=None,
        save_path=None, 
        particle=None,
        ctf=None,
        inverse_colour=True,
        return_features=False, 
        is_physical=False,
    ):
        if isinstance(clean_mrc, str):
            clean_mrc = read_image(clean_mrc)
        assert clean_mrc.ndim == 2, "Input image must be 2D"
        if is_physical:
            input = clean_mrc
            if isinstance(input, np.ndarray): 
                input = torch.from_numpy(input).float().to(self.device)
            if input.ndim == 2: 
                input = input.unsqueeze(0).unsqueeze(0)
        else:
            input = self.apply_physical_based_function(
                clean_mrc,
                weight_map=weight_map,
                particle=particle,
                ctf=ctf,
                inverse_colour=inverse_colour,
            )
        with torch.no_grad():
            outputs = self.model( 
                input,
                apply_ctf=False,
                apply_gaussian_noise=True,
                snr=0.1,
                apix=self.apix,
                nce_layers=[1,2,3,4,5],
                return_features=return_features,
            )
            if return_features: 
                input_features, output_features, synthetic, physical_based = outputs
            else:
                synthetic, physical_based = outputs
                input_features, output_features = None, None
        synthetic = synthetic[0,0,...].detach().cpu().numpy()
        physical_based = input[0,0,...].detach().cpu().numpy()
        synthetic_vmin, synthetic_vmax = synthetic.min(), synthetic.max()
        synthetic_micrograph, physical_based_micorgraph = synthetic, physical_based 
        if particle is not None:
            physical_based_particle = crop_particle_from_center(physical_based, particle.shape)
            synthetic_particle = crop_particle_from_center(synthetic, particle.shape)
            if save_path is not None:
                write_image(save_path[:-4] + ".particle.physical-based.png", physical_based_particle)
                write_image(save_path[:-4]+".particle.synthetic.png",synthetic_particle, synthetic_vmin, synthetic_vmax)
        else:
            physical_based_particle, synthetic_particle = None, None
        if save_path is not None:
            write_image(save_path[:-4] + ".micrograph.physical-based.png", physical_based_micorgraph)
            write_image(save_path[:-4]+".micrograph.synthetic.png", synthetic_micrograph, synthetic_vmin, synthetic_vmax)
        results_dict = {
            "physical_based_micorgraph": physical_based_micorgraph,
            "synthetic_micrograph": synthetic_micrograph,
            "physical_based_particle": physical_based_particle,
            "synthetic_particle": synthetic_particle,
            "genem_input_features": input_features,
            "genem_output_features": output_features,
        }
        return results_dict
        
    def gen_template(self, 
            m_len, 
            mask_ratio, 
            particles, 
            add_particle_one_by_one=False,
            save_name=None,
            mask=None, # initial mask for the micrograph, e.g. cvpr, eccv
            empty_center=False, # do not generate particles at the center of the micrograph
        ):
        p_num, p_len = particles.shape[:2]
        mask = np.zeros((m_len, m_len), dtype=bool) if mask is None else mask
        if empty_center:
            mask = paste_particle_at_center(mask, np.ones((int(p_len*1.5), int(p_len*1.5)), dtype=bool))
        micrograph = np.zeros((m_len+2*p_len, m_len+2*p_len), dtype=np.float32)
        mask_len = int(p_len * (1-mask_ratio))
        if add_particle_one_by_one: save_name_list = []
        for i in range(p_num):
            coords = np.where(mask == False)
            if coords[0].shape[0] == 0: break
            coords_x, coords_y = coords[0], coords[1]
            k = np.random.randint(0, coords_x.shape[0], 1)[0]
            cx, cy = coords_x[k], coords_y[k]
            radius = p_len//2
            micrograph[cx+p_len-radius:cx+p_len+(p_len-radius), cy+p_len-radius:cy+p_len+(p_len-radius)] += particles[i]
            mask_x_min, mask_x_max = max(0, cx-mask_len), min(m_len, cx+mask_len)
            mask_y_min, mask_y_max = max(0, cy-mask_len), min(m_len, cy+mask_len)
            mask[mask_x_min:mask_x_max, mask_y_min:mask_y_max] = True
            if add_particle_one_by_one and save_name is not None:
                save_mrc = save_name + f".{i+1}.mrc"
                write_image(save_mrc, micrograph[p_len:-p_len, p_len:-p_len])
                save_name_list.append(save_mrc)
        if add_particle_one_by_one: 
            return save_name_list
        micrograph = micrograph[p_len:-p_len, p_len:-p_len]
        mask = get_mask(micrograph)
        if save_name is not None:
            write_image(save_name + '.mrc', micrograph)
            write_image(save_name + '.png', micrograph)
            write_image(save_name + '.mask.png', mask)
        # return micrograph
        return_outputs = {
            "clean_mrc": micrograph,
            "mask": mask,
            "clean_mrc_path": save_name + '.mrc' if save_name is not None else None,
            "mask_path": save_name + '.mask.png' if save_name is not None else None,
        }
        return return_outputs