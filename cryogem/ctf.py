import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_safe_freqs(n_pixels, psize):
    freq_pix_1d = np.arange(-0.5, 0.5, 1 / n_pixels)
    freq_pix_1d_safe = freq_pix_1d[:n_pixels]
    x, y = np.meshgrid(freq_pix_1d_safe, freq_pix_1d_safe)
    rho = np.sqrt(x**2 + y**2)
    angles_rad = np.arctan2(y, x)
    freq_mag_2d = rho / psize
    return freq_mag_2d, angles_rad

def torch_compute_safe_freqs(n_pixels, psize):
    freq_pix_1d = torch.arange(-0.5, 0.5, 1 / n_pixels)
    freq_pix_1d_safe = freq_pix_1d[:n_pixels]
    x, y = torch.meshgrid(freq_pix_1d_safe, freq_pix_1d_safe)
    rho = torch.sqrt(x**2 + y**2)
    angles_rad = torch.atan2(y, x)
    freq_mag_2d = rho / psize
    return freq_mag_2d, angles_rad

def generate_random_ctf_params(batch_size, 
                                df_min=15000,
                                df_max=20000,
                                df_diff_min=100,
                                df_diff_max=500,
                                df_ang_min=0,
                                df_ang_max=360,
                                volt=300, 
                                cs=2.7, 
                                w=0.1, 
                                phase_shift=0,):
    dfs = np.random.uniform(low=df_min, high=df_max, size=(batch_size, 1, 1))
    df_diff = np.random.uniform(low=df_diff_min, high=df_diff_max, size=(batch_size, 1, 1))
    df_ang_deg = np.random.uniform(low=df_ang_min, high=df_ang_max, size=(batch_size, 1, 1))
    dfu = dfs - df_diff / 2
    dfv = dfs + df_diff / 2
    return dfu, dfv, df_ang_deg, np.ones((batch_size, 1, 1)) * volt, np.ones((batch_size, 1, 1)) * cs, np.ones((batch_size, 1, 1)) * w, np.ones((batch_size, 1, 1)) * phase_shift

def generate_sequential_ctf_params(batch_size,
                                    df_min=15000,
                                    df_max=20000,
                                    df_diff_min=100,
                                    df_diff_max=500,
                                    df_ang_min=0,
                                    df_ang_max=360,
                                    volt=300, 
                                    cs=2.7, 
                                    w=0.1, 
                                    phase_shift=0,):
    dfs = np.linspace(df_min, df_max, batch_size).reshape(-1, 1, 1)
    df_diff = np.random.uniform(low=df_diff_min, high=df_diff_max, size=(batch_size, 1, 1))
    df_ang_deg = np.random.uniform(low=df_ang_min, high=df_ang_max, size=(batch_size, 1, 1))
    # df_diff = np.array([(df_diff_max + df_diff_min) / 2] * batch_size, dtype=np.float32).reshape(-1, 1, 1)
    # df_ang_deg = np.array([(df_ang_max + df_ang_min) / 2] * batch_size, dtype=np.float32).reshape(-1, 1, 1)   
    dfu = dfs - df_diff / 2
    dfv = dfs + df_diff / 2
    return dfu, dfv, df_ang_deg, np.ones((batch_size, 1, 1)) * volt, np.ones((batch_size, 1, 1)) * cs, np.ones((batch_size, 1, 1)) * w, np.ones((batch_size, 1, 1)) * phase_shift


def compute_ctf(s, a, dfu, dfv, dfang_deg, kv, cs, w, phase=0, bf=0):
    s = s[None, ...] # add batch dimension
    a = a[None, ...] # add batch dimension
    kv = kv * 1e3
    cs = cs * 1e7
    lamb = 12.2643247 / np.sqrt(kv * (1.0 + kv * 0.978466e-6))

    dfang_deg = np.deg2rad(dfang_deg)
    def_avg = -(dfu + dfv) * 0.5
    def_dev = -(dfu - dfv) * 0.5
    k1 = np.pi / 2.0 * 2 * lamb
    k2 = np.pi / 2.0 * cs * lamb**3
    k3 = np.sqrt(1 - w**2)
    k4 = bf / 4.0  # B-factor, follows RELION convention.
    k5 = np.deg2rad(phase)  # Phase shift.
    s_2 = s**2
    s_4 = s_2**2
    dZ = def_avg + def_dev * (np.cos(2 * (a - dfang_deg)))
    gamma = (k1 * dZ * s_2) + (k2 * s_4) - k5
    ctf = -(k3 * np.sin(gamma) - w * np.cos(gamma))
    if bf != 0:  # Enforce envelope.
        ctf *= np.exp(-k4 * s_2)
    return ctf


def torch_compute_ctf(s, a, dfu, dfv, dfang_deg, kv, cs, w, phase=0, bf=0):
    s = s[None, ...] # add batch dimension
    a = a[None, ...] # add batch dimension
    kv = kv * 1e3
    cs = cs * 1e7
    lamb = 12.2643247 / torch.sqrt(kv * (1.0 + kv * 0.978466e-6))

    dfang_deg = torch.deg2rad(dfang_deg)
    def_avg = -(dfu + dfv) * 0.5
    def_dev = -(dfu - dfv) * 0.5
    k1 = np.pi / 2.0 * 2 * lamb
    k2 = np.pi / 2.0 * cs * lamb**3
    k3 = torch.sqrt(1 - w**2)
    k4 = bf / 4.0  # B-factor, follows RELION convention.
    k5 = torch.deg2rad(phase)  # Phase shift.
    s_2 = s**2
    s_4 = s_2**2
    dZ = def_avg + def_dev * (torch.cos(2 * (a - dfang_deg)))
    gamma = (k1 * dZ * s_2) + (k2 * s_4) - k5
    ctf = -(k3 * torch.sin(gamma) - w * torch.cos(gamma))
    if bf != 0:  # Enforce envelope.
        ctf *= torch.exp(-k4 * s_2)
    return ctf

def fftn_center(img):
    return np.fft.fftshift(np.fft.fftn(img))


def fft2_center(img):
    return np.fft.fftshift(np.fft.fft2(img), axes=(-2, -1))


def ifft2_center(img):
    return np.fft.ifft2(np.fft.ifftshift(img, axes=(-2, -1)))


def torch_fft2_center(img):
    return torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))


def torch_ifft2_center(img):
    return torch.fft.ifft2(torch.fft.ifftshift(img, dim=(-2, -1)))