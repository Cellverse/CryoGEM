import mrcfile, os, torch
import numpy as np
import torch.nn as nn

from cryogem.utils_gen_data import generate_hetero_data_given_conform_pose
from cryogem.lie_tools import generate_y_rotations, quaternion2Rmatrix, RotateProject

import logging

logger = logging.getLogger(__name__)

class Protein_volume(nn.Module):

    def __init__(self, mode, device, vol_path=None, zxy=False, drgn_dir=None, drgn_epoch=None, zfile=None):
        super(Protein_volume, self).__init__()
        self.mode = mode
        self.device = device
        assert mode in ["homo", "hetero"], "please check mode."
        if self.mode == 'homo':
            logger.info(f"Protein: {mode}, {vol_path}")
            self.homo_build(vol_path, zxy=zxy)
        else:
            logger.info(f"Protein: {drgn_dir}, {drgn_epoch}, {zfile}")
            self.hetero_build(drgn_dir, drgn_epoch, zfile)
    
    def gen_rotations(self, n_rotations, symmetry="C1"):
        rotations = generate_y_rotations(n_rotations, 2*np.pi/float(symmetry[-1]))
        if self.mode == "hetero": # 10345 fixed initial pose
            quat = np.array([ 0.6590068, -0.28052469, 0.41374563, 0.56198798])
            rot = np.transpose(quaternion2Rmatrix(quat[np.newaxis,...]), axes=(2, 0, 1))
            initial_rots = np.tile(rot,(n_rotations, 1,1)).astype(np.float32)
            rotations = generate_y_rotations(n_rotations, 2*np.pi)
            rotations = np.matmul(rotations, initial_rots)
        rotations = rotations.astype(np.float32)
        return rotations
    
    def gen_same_rotations(self, n_rotations):
        quat = np.array([ 0.6590068, -0.28052469, 0.41374563, 0.56198798])
        rot = np.transpose(quaternion2Rmatrix(quat[np.newaxis,...]), axes=(2, 0, 1))
        rotations = np.tile(rot,(n_rotations, 1,1)).astype(np.float32)
        return rotations
    
    def gen_z_vals(self, num, linear=False):
        z_vals = self.z_vals
        gen_z_vals = []
        if not linear:
            for i in range(num):
                z_type = np.random.randint(0, z_vals.shape[0]-1, 1)
                w = np.random.rand(1)
                gen_z_vals.append(
                    (1-w)*z_vals[z_type]+w*z_vals[z_type+1]
                )
        else:
            w = np.linspace(0, 1, num//(z_vals.shape[0]-1)).reshape(-1, 1)
            for i in range(z_vals.shape[0]-1):
                gen_z_vals.append(
                    (1-w) * z_vals[i:i+1] + w * z_vals[i+1:i+2]
                )
        gen_z_vals = np.concatenate(gen_z_vals, axis=0).astype(np.float32)
        return gen_z_vals

    def homo_build(self, vol_path, zxy=False):
        vol = mrcfile.open(vol_path, permissive=True).data.copy()
        if zxy: vol = np.transpose(vol, (2,0,1))
        sidelen = vol.shape[0]
        self.vol = torch.tensor(vol).view(1, 1, sidelen, sidelen, sidelen).to(self.device)
        self.rotate_project = RotateProject(sidelen).to(self.device)
    
    def hetero_build(self, drgn_dir, drgn_epoch, zfile):
        self.z_vals = np.loadtxt(zfile)
        self.vol = lambda rotations, z_vals: generate_hetero_data_given_conform_pose(
            drgn_dir, drgn_epoch, rotations.shape[0], z_vals, rotations, downsample_boxsize=None, device=self.device
        )

    def forward(self, rotations, double=False, z_vals=None):
        N = rotations.shape[0]
        if self.mode == "homo":
            particles = []
            for i in range(N):
            # for i in tqdm(range(N), desc="Generating projections..."):
                particle = self.rotate_project(
                    self.vol.expand(1, -1, -1, -1, -1), 
                    torch.tensor(rotations[i:i+1,...]).float().to(self.device)
                ).detach().cpu().numpy()[0, ...]
                particles.append(particle)
            particles = np.stack(particles, axis=0)
        else:
            if z_vals is None: z_vals = np.array([self.z_vals[self.z_vals.shape[0]//2,...]]*rotations.shape[0]).astype(np.float32)
            particles = self.vol(
                rotations,
                z_vals
            )
        if double:
            particles = np.concatenate([particles, particles[::-1]], axis=0)
        return particles
    

exp_abinitio_volumes_dir = "testing/data/exp_abinitio_volumes"
volume_dict = { 
    "Proteasome(10025)": {
        "vol_path": os.path.join(exp_abinitio_volumes_dir, "densitymap.10025.64.mrc"),
        "zxy": False,
    },
    "Ribosome(10028)":  {
        "vol_path": os.path.join(exp_abinitio_volumes_dir, "densitymap.10028.90.mrc"),
        "zxy": True,
    },
    "Integrin(10345)": {
        "drgn_dir": os.path.join(exp_abinitio_volumes_dir, "10345_neural_volume/drgn_result"),
        "drgn_epoch": 49,
        "zfile": os.path.join(exp_abinitio_volumes_dir, "10345_neural_volume/drgn_result/analyze.49/pc2/z_values.txt"),
    },
    "PhageMS2(10075)": {
        "vol_path": os.path.join(exp_abinitio_volumes_dir, "densitymap.10075.112.mrc"),
        "zxy": False,
    },
    "HumanBAF(10590)": {
        "vol_path": os.path.join(exp_abinitio_volumes_dir, "densitymap.10590.84.mrc"),
        "zxy": False,
    },
}

def get_volume(dataset, device):
    mode = "homo" if dataset != "Integrin(10345)" else "hetero"
    args = volume_dict[dataset]
    prot_vol = Protein_volume(
        mode, device,
        **args
    )
    return prot_vol
