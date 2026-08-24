"""
Signal-based loss functions for ECG reconstruction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTLoss(nn.Module):
    """Multi-resolution STFT loss for spectral fidelity."""
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_lengths=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
    
    def forward(self, pred, target):
        loss = 0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            # Compute STFT for each resolution
            pred_stft = torch.stft(pred.view(-1, pred.size(-1)), n_fft=fft_size, hop_length=hop_size, 
                                   win_length=win_length, return_complex=True, center=True)
            target_stft = torch.stft(target.view(-1, target.size(-1)), n_fft=fft_size, hop_length=hop_size,
                                     win_length=win_length, return_complex=True, center=True)
            
            # Magnitude spectrogram loss
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            loss += F.l1_loss(pred_mag, target_mag)
            
            # Log magnitude loss
            pred_log = torch.log(pred_mag + 1e-7)
            target_log = torch.log(target_mag + 1e-7)
            loss += F.l1_loss(pred_log, target_log)
        
        return loss / len(self.fft_sizes)


class BandPowerLoss(nn.Module):
    """Frequency band power matching loss."""
    def __init__(self, sr=500, bands=[(0, 5), (5, 50), (50, 100), (100, 240)]):
        super().__init__()
        self.sr = sr
        self.bands = bands
    
    def forward(self, pred, target):
        loss = 0
        n_fft = 1024
        
        pred_flat = pred.view(-1, pred.size(-1))
        target_flat = target.view(-1, target.size(-1))
        
        pred_fft = torch.fft.rfft(pred_flat, n=n_fft)
        target_fft = torch.fft.rfft(target_flat, n=n_fft)
        
        pred_psd = torch.abs(pred_fft) ** 2
        target_psd = torch.abs(target_fft) ** 2
        
        freqs = torch.fft.rfftfreq(n_fft, d=1/self.sr).to(pred.device)
        
        for low, high in self.bands:
            mask = (freqs >= low) & (freqs < high)
            pred_power = pred_psd[:, mask].mean(dim=1)
            target_power = target_psd[:, mask].mean(dim=1)
            loss += F.l1_loss(torch.log(pred_power + 1e-7), torch.log(target_power + 1e-7))
        
        return loss / len(self.bands)


class CompositeLoss(nn.Module):
    """Composite loss combining MSE, STFT, and BandPower."""
    def __init__(self, mse_weight=1.0, stft_weight=0.1, band_weight=0.05):
        super().__init__()
        self.mse_weight = mse_weight
        self.stft_weight = stft_weight
        self.band_weight = band_weight
        self.stft_loss = STFTLoss()
        self.band_loss = BandPowerLoss()
    
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        stft = self.stft_loss(pred, target)
        band = self.band_loss(pred, target)
        return self.mse_weight * mse + self.stft_weight * stft + self.band_weight * band
