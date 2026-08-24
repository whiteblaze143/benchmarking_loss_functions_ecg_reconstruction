
import numpy as np
import torch
import wfdb
import os
from scipy.signal import resample_poly

def load_nstdb_noise(noise_type, data_dir="data/mit-bih-noise-stress-test-database-1.0.0", target_fs=500):
    """
    Load specific noise record from NSTDB.
    noise_type: 'bw', 'em', 'ma'
    """
    # Record names in NSTDB: 'bw', 'em', 'ma'
    record_path = os.path.join(data_dir, noise_type)
    
    try:
        # Load signal
        # NSTDB is 360Hz. We must resample to 500Hz.
        record = wfdb.rdrecord(record_path)
        sig = record.p_signal[:, 0] # Use channel 0
        fs_orig = record.fs
        
        if fs_orig != target_fs:
            # Resample
            # GCD of 360 and 500 is 20.
            # 360/20 = 18. 500/20 = 25.
            # Up 25, Down 18.
            sig = resample_poly(sig, 25, 18)
            
        return sig
    except Exception as e:
        print(f"Error loading {noise_type}: {e}")
        # Fallback: Simulated Noise (Approximation)
        # BW: Low frequency drift
        # MA: High frequency noise
        # EM: Random step/walk
        print(f"Falling back to simulated {noise_type} noise.")
        L = 30 * 500 * 60 # 30 mins
        if noise_type == 'bw':
            # Sinusoidal drift mixture
            t = np.linspace(0, 30*60, L)
            sig = np.sin(2*np.pi*0.2*t) + 0.5*np.sin(2*np.pi*0.6*t)
        elif noise_type == 'ma':
            # Gaussian noise high freq
            sig = np.random.normal(0, 1, L)
        else:
            # White noise
            sig = np.random.normal(0, 1, L)
            
        return sig

def mix_ecg_with_nstdb_noise(ecg_signal, noise_signal, target_snr_db):
    """
    Mix ECG with Noise at target SNR.
    Args:
        ecg_signal: [L] numpy array
        noise_signal: [L_noise] numpy array (source noise bank)
        target_snr_db: float
    """
    L = len(ecg_signal)
    
    # 1. Random crop of noise
    if len(noise_signal) <= L:
        # Loop noise if too short (unlikely for NSTDB which is 30 mins)
        noise_crop = np.resize(noise_signal, L)
    else:
        start_idx = np.random.randint(0, len(noise_signal) - L)
        noise_crop = noise_signal[start_idx : start_idx + L]
        
    # 2. Calculate Energies
    # Remove mean to measure variance/energy properly
    ecg_clean = ecg_signal - np.mean(ecg_signal)
    noise_clean = noise_crop - np.mean(noise_crop)
    
    params_ecg = np.sum(ecg_clean ** 2)
    params_noise = np.sum(noise_clean ** 2)
    
    if params_noise == 0:
        return ecg_signal, noise_crop # Avoid div/0
        
    # 3. Scale Noise
    # SNR_db = 10 * log10(P_signal / P_noise)
    # P_noise_target = P_signal / (10 ** (SNR_db / 10))
    # We scale noise amplitude by factor alpha.
    # New P_noise = alpha^2 * old_P_noise
    # alpha = sqrt(P_noise_target / old_P_noise)
    
    target_noise_power = params_ecg / (10 ** (target_snr_db / 10))
    alpha = np.sqrt(target_noise_power / params_noise)
    
    scaled_noise = noise_clean * alpha
    
    # 4. Mix
    c_ecg = ecg_signal + scaled_noise
    
    return c_ecg, scaled_noise

class StressTester:
    def __init__(self, nstdb_dir="data/mit-bih-noise-stress-test-database-1.0.0", fs=500):
        self.noise_types = ['bw', 'em', 'ma']
        self.noises = {}
        for nt in self.noise_types:
            print(f"Loading {nt} noise...")
            self.noises[nt] = load_nstdb_noise(nt, nstdb_dir, fs)
            
    def get_corrupted_batch(self, x_batch, noise_type, snr):
        """
        x_batch: [B, 1, L] tensor
        Returns: [B, 1, L] tensor (noisy)
        """
        noise_source = self.noises.get(noise_type)
        if noise_source is None:
            return x_batch
            
        x_noisy_list = []
        for i in range(x_batch.shape[0]):
            sig = x_batch[i, 0].cpu().numpy()
            noisy_sig, _ = mix_ecg_with_nstdb_noise(sig, noise_source, snr)
            x_noisy_list.append(noisy_sig)
            
        return torch.tensor(np.array(x_noisy_list), dtype=torch.float32).unsqueeze(1).to(x_batch.device)

