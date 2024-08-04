import logging
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from kornia.geometry.transform import warp_affine3d


logger = logging.getLogger(__name__)

def random_quaternion(N):
    u = np.random.uniform(0, 1, size=(N,))
    v = np.random.uniform(0, 1, size=(N,))
    w = np.random.uniform(0, 1, size=(N,))
    q = np.stack([np.sqrt(1 - u) * np.sin(2*np.pi*v), 
                  np.sqrt(1 - u) * np.cos(2*np.pi*v), 
                  np.sqrt(u) * np.sin(2*np.pi*w),
                  np.sqrt(u) * np.cos(2*np.pi*w)], axis=-1)
    return q

    
def quaternion2Rmatrix(q):
    qi = q[:, 0]
    qj = q[:, 1]
    qk = q[:, 2]
    qr = q[:, 3]
    r11 = 1 - 2*(qj**2 + qk**2)
    r12 = 2*(qi*qj - qk*qr)
    r13 = 2*(qi*qk + qj*qr)
    r21 = 2*(qi*qj + qk*qr)
    r22 = 1 - 2*(qi**2 + qk**2)
    r23 = 2*(qj*qk - qi*qr)
    r31 = 2*(qi*qk - qj*qr)
    r32 = 2*(qj*qk + qi*qr)
    r33 = 1 - 2*(qi**2 + qj**2)
    R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    return R

def random_theta_phi(N, symmetry="C1"):
    assert len(symmetry) == 2, "symmetry format: C/D 1,2,3,4,5,6,7"
    assert symmetry[0] in ["C", "D"], "symmetry format: C/D 1,2,3,4,5,6,7"
    theta_max = 2*np.pi/float(symmetry[1])
    phi_max = np.pi if symmetry[0] == "C" else np.pi/2
    
    theta = np.random.uniform(0, theta_max, size=(N,))
    phi = np.random.uniform(0, phi_max, size=(N,))
    
    return theta, phi

def theta_phi2Rmatrix(theta, phi):
    # theta: [N]
    # phi: [N]
    R = np.zeros((len(theta), 3, 3))

    for i in range(len(theta)):
        Rz = np.array([
            [np.cos(theta[i]), -np.sin(theta[i]), 0],
            [np.sin(theta[i]), np.cos(theta[i]), 0],
            [0, 0, 1]
        ])

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi[i]), -np.sin(phi[i])],
            [0, np.sin(phi[i]), np.cos(phi[i])]
        ])

        R[i] = np.dot(Rz, Rx)
        
    return R
    
class AffineGenerator(Dataset):
    def __init__(self, n_projections, symmetry="C1"):
        if symmetry == "C1": 
            rotations = np.transpose(quaternion2Rmatrix(random_quaternion(n_projections)), axes=(2, 0, 1))
        else:
            rotations = theta_phi2Rmatrix(*random_theta_phi(n_projections, symmetry=symmetry))
        self.rotations = torch.tensor(rotations).float()
        self.n_projections = n_projections
        
    def __len__(self):
        return self.n_projections
    
    def __getitem__(self, idx):
        return self.rotations[idx]

def angles2Rmatrix(angles: list):
    Rs = []
    for angle in angles:
        # rotation around y axis
        R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        Rs.append(R)
    return np.stack(Rs, axis=0)

def angles2Rmatrix(angles: list):
    Rs = []
    for angle in angles:
        # rotation around y axis
        R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        Rs.append(R)
    return np.stack(Rs, axis=0)


class AffineUniformGenerator(Dataset):
    def __init__(self, n_projections):
        # angles = np.deg2rad(np.random.uniform(-60, 60, n_projections) + 90)
        angles = np.deg2rad(np.linspace(-60, 60, n_projections) + 90)
        
        rotations = angles2Rmatrix(angles)
        self.rotations = torch.tensor(rotations).float()
        self.n_projections = n_projections

    def __len__(self):
        return self.n_projections
    
    def __getitem__(self, idx):
        return self.rotations[idx]
    

class RotateProject(nn.Module):
    def __init__(self, sidelen):
        super(RotateProject, self).__init__()
        self.sidelen = sidelen

    def forward(self, vol, rotations):
        dv = rotations.device
        sidelen = vol.shape[-1]
        rotated_offset = sidelen / 2 * (torch.ones(3, 1, device=dv) - torch.matmul(rotations, torch.ones(3, 1, device=dv)))
        warp_affine = torch.cat([rotations, rotated_offset], axis=-1)
        warped_vols = warp_affine3d(vol, warp_affine, dsize=(sidelen, sidelen, sidelen), flags='bilinear', padding_mode='zeros')
        projections = warped_vols.sum(2).squeeze(1)
        return projections

def rotation_matrix_z(theta):
    """Returns a 3x3 rotation matrix for a rotation of theta radians about the z-axis."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,             0,             1]
    ])

def rotation_matrix_y(theta):
    """Returns a 3x3 rotation matrix for a rotation of theta radians about the y-axis."""
    return np.array([
        [np.cos(theta),  0, np.sin(theta)],
        [0,              1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
def rotation_matrix_x(theta):
    """Returns a 3x3 rotation matrix for a rotation of theta radians about the y-axis."""
    return np.array([
        [1,  0, 0],
        [0,  np.cos(theta),  np.sin(theta)],
        [0, -np.sin(theta),  np.cos(theta)]
    ])

def generate_spiral_rotations(num_steps, max_theta):
    """Generates a sequence of rotation matrices that simulate a spiral motion."""
    rotations = []
    for i in range(num_steps):
        # Linearly interpolate the angle from 0 to max_theta
        theta_z = max_theta * float(i) / num_steps
        theta_y = np.pi / 2 * float(i) / num_steps  # Elevate the view angle

        # Create the combined rotation matrix
        rotation_z = rotation_matrix_z(theta_z)
        rotation_y = rotation_matrix_y(theta_y)
        combined_rotation = np.dot(rotation_y, rotation_z)

        rotations.append(combined_rotation)

    rotations = np.stack(rotations, axis=0)
    return rotations

def generate_z_rotations(num_steps, max_theta):
    """Generates a sequence of rotation matrices for rotations about the z-axis."""
    rotations = []
    for i in range(num_steps):
        # Linearly interpolate the angle from 0 to max_theta
        theta = max_theta * float(i) / num_steps

        # Create the rotation matrix
        rotation = rotation_matrix_z(theta)
        rotations.append(rotation)

    rotations = np.stack(rotations, axis=0)
    return rotations

def generate_y_rotations(num_steps, max_theta):
    """Generates a sequence of rotation matrices for rotations about the y-axis."""
    rotations = []
    for i in range(num_steps):
        # Linearly interpolate the angle from 0 to max_theta
        theta = max_theta * float(i) / num_steps

        # Create the rotation matrix
        rotation = rotation_matrix_y(theta)
        rotations.append(rotation)

    rotations = np.stack(rotations, axis=0).astype(np.float32)
    return rotations

def generate_x_rotations(num_steps, max_theta):
    """Generates a sequence of rotation matrices for rotations about the y-axis."""
    rotations = []
    for i in range(num_steps):
        # Linearly interpolate the angle from 0 to max_theta
        theta = max_theta * float(i) / num_steps

        # Create the rotation matrix
        rotation = rotation_matrix_x(theta)
        rotations.append(rotation)

    rotations = np.stack(rotations, axis=0).astype(np.float32)
    return rotations