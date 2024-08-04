from cryogem.inference.config_infer import dataset_list, template_mask_ratio_dict, template_particle_num_dict
from cryogem.cryogem import CryoGEM
from cryogem.inference.utils_infer import gen_video
from cryogem.lie_tools import generate_y_rotations
from cryogem.ctf import generate_sequential_ctf_params, compute_safe_freqs, compute_ctf

import logging, argparse, os
import numpy as np 
from tqdm import tqdm 

logger = logging.getLogger(__name__)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--use_default", action="store_true", help="use default setting")
    parser.add_argument("--dataset", type=str, default="Ribosome(10028)", choices=dataset_list, help="default setting list")
    # cryogem
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
    parser.add_argument("--save_dir", type=str, default="save_images/video/gallery/Ribosome(10028)")
    parser.add_argument("--device", type=str, default="cuda:0", help="device, e.g. cpu, cuda:0, ...")
    parser.add_argument("--df_min", type=float, default=-20000, help="Minimum defocus value")
    parser.add_argument("--df_max", type=float, default=20000, help="Maximum defocus value")
    parser.add_argument("--size", type=int, default=1024, help="Micrograph size, [size size]")
    #! selectors 
    parser.add_argument("--pose", action="store_true", help="if apply, generate varied pose particles")
    parser.add_argument("--deform", action="store_true", help="if apply, generate varied deformation particles")
    parser.add_argument("--defocus", action="store_true", help="if apply, generate varied defocus particles and micrographs")
    
    parser.add_argument("--n_frames", type=int, default=120, help="Number of frames to generate for vaired pose or deformation")
    parser.add_argument("--n_particles", type=int, default=100, help="Number of particles to generate")
    parser.add_argument("--particle_collapse_ratio", type=float, default=0.6, help="Particle collapse ratio")
    parser.add_argument("--apply_mask", action="store_true", help="if apply, generate particles where the mask is zero (black).")
    parser.add_argument("--mask_path", type=str, default="testing/data/manual_created_mask/eccv.png", help="mask path")
    return parser

def main(args):
    if args.deform:
        args.dataset = "Integrin(10345)"
        args.use_default = True
        args.save_dir = "save_images/video/gallery/Integrin(10345)"
        
    if args.use_default:
        cryogem = CryoGEM(dataset=args.dataset, device=args.device)
        n_particles = template_particle_num_dict[args.dataset]
        particle_collapse_ratio = template_mask_ratio_dict[args.dataset]
    else:
        cryogem = CryoGEM(
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
    
    # first generate clean template
    rotations = cryogem.get_vol().gen_rotations(n_particles)
    particles = cryogem.get_vol()(rotations)
    clean_mrc = cryogem.gen_template(
        size,
        particle_collapse_ratio,
        particles,
        empty_center=True
    )["clean_mrc"]
    
    # varied pose
    if args.pose:
        save_dir = os.path.join(args.save_dir, "varied_pose.particles")
        rotations = generate_y_rotations(args.n_frames, 2*np.pi)
        particles = cryogem.get_vol()(rotations)
        physical_baseds, synthetics = [], []
        for i in tqdm(range(args.n_frames), desc="Varied pose (particles) ..."):
            outputs = cryogem.add_noise(
                clean_mrc.copy(),
                save_path=os.path.join(save_dir, f"{i:03d}.png"),
                particle=particles[i],
                inverse_colour=True,
            )
            physical_baseds.append(outputs["physical_based_particle"])
            synthetics.append(outputs["synthetic_particle"])
        physical_baseds = np.array(physical_baseds)
        synthetics = np.array(synthetics)
        gen_video(physical_baseds, os.path.join(args.save_dir, "varied_pose.particles.physical_based.mp4"))
        gen_video(synthetics, os.path.join(args.save_dir, "varied_pose.particles.synthetic.mp4"))
        
    # varied deformation
    if args.deform:
        save_dir = os.path.join(args.save_dir, "varied_deform.particles")
        rotations = cryogem.get_vol().gen_same_rotations(n_particles)
        z_vals = cryogem.get_vol().gen_z_vals(n_particles, linear=True)
        particles = cryogem.get_vol()(rotations, z_vals=z_vals)
        for i in tqdm(range(args.n_frames), desc="Varied deformation (particles) ..."):
            outputs = cryogem.add_noise(
                clean_mrc.copy(),
                save_path=os.path.join(save_dir, f"{i:03d}.png"),
                particle=particles[i],
                inverse_colour=True,
            )
            physical_baseds.append(outputs["physical_based_particle"])
            synthetics.append(outputs["synthetic_particle"])
        physical_baseds = np.array(physical_baseds)
        synthetics = np.array(synthetics)
        gen_video(physical_baseds, os.path.join(args.save_dir, "varied_deform.particles.physical_based.mp4"))
        gen_video(synthetics, os.path.join(args.save_dir, "varied_deform.particles.synthetic.mp4"))
        
    # varied defocus
    if args.defocus:
        save_dir = os.path.join(args.save_dir, "varied_defocus")
        rotations = cryogem.get_vol().gen_rotations(1)
        particles = cryogem.get_vol()(rotations)
        ctf_params = generate_sequential_ctf_params(args.n_frames, df_min=args.df_min, df_max=args.df_max)
        freqs_mag, angles_rad = compute_safe_freqs(size, cryogem.apix)
        ctfs = compute_ctf(freqs_mag, angles_rad, *ctf_params).reshape(args.n_frames, 1, 1024, 1024)
        physical_baseds_mic, synthetics_mic = [], []
        physical_baseds_par, synthetics_par = [], []
        for i in tqdm(range(args.n_frames), desc="Varied defocus (micrographs)..."):
            outputs = cryogem.add_noise(
                clean_mrc.copy(),
                save_path=os.path.join(save_dir, f"{i:03d}.png"),
                ctf=ctfs[i],
                particle=particles[0],
                inverse_colour=True,
            )
            physical_baseds_mic.append(outputs["physical_based_micorgraph"])
            synthetics_mic.append(outputs["synthetic_micrograph"])
            physical_baseds_par.append(outputs["physical_based_particle"])
            synthetics_par.append(outputs["synthetic_particle"])
        physical_baseds_mic = np.array(physical_baseds_mic)
        synthetics_mic = np.array(synthetics_mic)
        physical_baseds_par = np.array(physical_baseds_par)
        synthetics_par = np.array(synthetics_par)
        gen_video(physical_baseds_mic, os.path.join(args.save_dir, "varied_defocus.micrograph.physical_based.mp4"))
        gen_video(synthetics_mic, os.path.join(args.save_dir, "varied_defocus.micrograph.synthetic.mp4"))
        gen_video(physical_baseds_par, os.path.join(args.save_dir, "varied_defocus.particles.physical_based.mp4"))
        gen_video(synthetics_par, os.path.join(args.save_dir, "varied_defocus.particles.synthetic.mp4"))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())



