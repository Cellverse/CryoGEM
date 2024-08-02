from genem.inference.config_infer import dataset_list, template_mask_ratio_dict, template_particle_num_dict
from genem.genem import GenEM
from genem.inference.utils_infer import gen_video

import logging, argparse, os
import numpy as np 
from tqdm import tqdm 

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
    parser.add_argument("--save_dir", type=str, default="save_images/video/eccv/Ribosome(10028)")
    parser.add_argument("--device", type=str, default="cuda:0", help="device, e.g. cpu, cuda:0, ...")
    parser.add_argument("--df_min", type=float, default=15000, help="Minimum defocus value")
    parser.add_argument("--df_max", type=float, default=20000, help="Maximum defocus value")
    parser.add_argument("--size", type=int, default=1024, help="Micrograph size, [size size]")
    parser.add_argument("--n_particles", type=int, default=100, help="Number of particles to generate")
    parser.add_argument("--particle_collapse_ratio", type=float, default=0.6, help="Particle collapse ratio")
    parser.add_argument("--apply_mask", action="store_true", help="if apply, generate particles where the mask is zero (black).")
    parser.add_argument("--mask_path", type=str, default="testing/data/manual_created_mask/eccv.png", help="mask path")
    return parser

def main(args):
    
    if args.use_default:
        genem = GenEM(dataset=args.dataset, device=args.device)
        n_particles = template_particle_num_dict[args.dataset]
        particle_collapse_ratio = template_mask_ratio_dict[args.dataset]
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
    logger.info(
        f"Generate sequences of micrograph ({size}x{size}), from 1 ~ {n_particles} particles and particle collapse ratio {particle_collapse_ratio}."
    )
    
    # first generate sequences of clean micrographs
    save_dir = os.path.join(args.save_dir, 'clean')
    rotations = genem.get_vol().gen_rotations(n_particles)
    particles = genem.get_vol()(rotations)
    clean_mic_save_name_list = genem.gen_template(
        size, 
        particle_collapse_ratio,
        particles,
        add_particle_one_by_one=True,
        save_name = os.path.join(
            save_dir, "pnum"
        ),
        empty_center=False
    )
    # next add noise to clean micrographs
    save_dir = os.path.join(args.save_dir, 'noisy')
    physical_baseds, synthetics = [], []
    for save_name in tqdm(clean_mic_save_name_list, "Add noise to clean micrographs..."):
        save_path = os.path.join(save_dir, os.path.basename(save_name)[:-4]+'.png')
        outputs = genem.add_noise(
                save_name,
                save_path=save_path,
                inverse_colour=True,
        )
        physical_baseds.append(outputs["physical_based_micorgraph"])
        synthetics.append(outputs["synthetic_micrograph"])
    
    physical_baseds = np.array(physical_baseds)
    synthetics = np.array(synthetics)
        
    gen_video(physical_baseds, os.path.join(args.save_dir, "physical_based.mp4"))
    gen_video(synthetics, os.path.join(args.save_dir, "synthetic.mp4"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())


