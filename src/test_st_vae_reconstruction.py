"""
Test ST-VAE reconstruction quality on normal and abnormal ECGs
Visualize time domain and frequency domain reconstructions
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_processing.dataset import ECGDataset
from models.st_vae import ST_VAE

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

def plot_reconstruction_comparison(model, dataset, device, num_samples=3, save_path=None):
    """Plot time and frequency domain reconstruction for normal and abnormal ECGs"""
    model.eval()
    
    # Get samples with correct length (1000)
    valid_samples = []
    for i in range(min(100, len(dataset))):  # Check first 100 samples
        x = dataset[i]
        # ECGDataset returns (T, C) shape, we need (T, 12)
        if x.shape[0] >= 1000 and x.shape[1] == 12:
            # Trim or pad to exactly 1000
            if x.shape[0] > 1000:
                x = x[:1000, :]
            valid_samples.append(x)
            if len(valid_samples) >= num_samples:
                break
    
    if len(valid_samples) < num_samples:
        print(f"Warning: Only found {len(valid_samples)} samples with length 1000")
        num_samples = len(valid_samples)
    
    x = torch.stack(valid_samples).to(device)
    
    with torch.no_grad():
        # Get reconstruction
        x_mean, x_logvar, mu, logvar, _, _ = model(x)
        
        # Compute reconstruction error
        recon_error = torch.mean((x - x_mean)**2, dim=(1, 2)).cpu().numpy()
        
        # Get frequency spectra
        x_fft = torch.fft.rfft(x.transpose(1, 2), dim=-1).abs()
        recon_fft = torch.fft.rfft(x_mean.transpose(1, 2), dim=-1).abs()
    
    # Move to CPU
    x = x.cpu().numpy()
    x_mean = x_mean.cpu().numpy()
    x_fft = x_fft.cpu().numpy()
    recon_fft = recon_fft.cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(20, 4 * num_samples))
    
    for i in range(num_samples):
        # Plot 3 representative leads: I (0), II (1), V1 (6)
        leads_to_plot = [0, 1, 6]
        lead_names = ['Lead I', 'Lead II', 'Lead V1']
        
        for idx, (lead, lead_name) in enumerate(zip(leads_to_plot, lead_names)):
            # Time domain reconstruction
            ax_time = plt.subplot(num_samples, 6, i * 6 + idx * 2 + 1)
            time = np.arange(x.shape[1])
            ax_time.plot(time, x[i, :, lead], 'b-', linewidth=1.5, label='Original', alpha=0.7)
            ax_time.plot(time, x_mean[i, :, lead], 'r--', linewidth=1.5, label='Reconstructed', alpha=0.7)
            ax_time.set_xlabel('Time (samples)')
            ax_time.set_ylabel('Amplitude (mV)')
            ax_time.set_title(f'Sample {i+1} - {lead_name} (Time Domain)\nError: {recon_error[i]:.4f}')
            ax_time.legend(loc='upper right', fontsize=8)
            ax_time.grid(True, alpha=0.3)
            
            # Frequency domain reconstruction
            ax_freq = plt.subplot(num_samples, 6, i * 6 + idx * 2 + 2)
            freq = np.fft.rfftfreq(x.shape[1], d=1/500)  # Assuming 500 Hz sampling
            ax_freq.plot(freq[:100], x_fft[i, lead, :100], 'b-', linewidth=1.5, label='Original', alpha=0.7)
            ax_freq.plot(freq[:100], recon_fft[i, lead, :100], 'r--', linewidth=1.5, label='Reconstructed', alpha=0.7)
            ax_freq.set_xlabel('Frequency (Hz)')
            ax_freq.set_ylabel('Magnitude')
            ax_freq.set_title(f'{lead_name} (Frequency Domain)')
            ax_freq.legend(loc='upper right', fontsize=8)
            ax_freq.grid(True, alpha=0.3)
            ax_freq.set_xlim([0, 50])  # Focus on 0-50 Hz (most important for ECG)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reconstruction comparison to {save_path}")
    
    plt.show()
    plt.close()
    
    return recon_error.mean()


def compute_reconstruction_statistics(model, dataset, device, num_samples=100):
    """Compute detailed reconstruction statistics"""
    model.eval()
    
    all_errors = []
    all_kl_divs = []
    all_time_errors = []
    all_freq_errors = []
    
    samples_processed = 0
    
    # Filter samples with correct length
    for i in range(len(dataset)):
        if samples_processed >= num_samples:
            break
        
        x = dataset[i]
        # ECGDataset returns (T, C) shape
        if x.shape[0] < 1000 or x.shape[1] != 12:
            continue
        
        # Trim to exactly 1000 if longer
        if x.shape[0] > 1000:
            x = x[:1000, :]
        
        x = x.unsqueeze(0).to(device)
        
        with torch.no_grad():
            x_mean, x_logvar, mu, logvar, _, _ = model(x)
            
            # Time domain error
            time_error = torch.mean((x - x_mean)**2, dim=(1, 2))
            
            # Frequency domain error
            x_fft = torch.fft.rfft(x.transpose(1, 2), dim=-1).abs()
            recon_fft = torch.fft.rfft(x_mean.transpose(1, 2), dim=-1).abs()
            freq_error = torch.mean((x_fft - recon_fft)**2, dim=(1, 2))
            
            # KL divergence
            mu_flat = mu[:, 0, :]
            logvar_flat = logvar[:, 0, :]
            kl = -0.5 * torch.sum(1 + logvar_flat - mu_flat.pow(2) - logvar_flat.exp(), dim=1)
            
            all_errors.append(time_error.cpu().numpy())
            all_time_errors.append(time_error.cpu().numpy())
            all_freq_errors.append(freq_error.cpu().numpy())
            all_kl_divs.append(kl.cpu().numpy())
            
            samples_processed += 1
    
    all_errors = np.concatenate(all_errors)
    all_time_errors = np.concatenate(all_time_errors)
    all_freq_errors = np.concatenate(all_freq_errors)
    all_kl_divs = np.concatenate(all_kl_divs)
    
    print("\n" + "="*70)
    print("RECONSTRUCTION STATISTICS")
    print("="*70)
    print(f"Number of samples: {len(all_errors)}")
    print(f"\nTime Domain Reconstruction Error:")
    print(f"  Mean: {all_time_errors.mean():.6f}")
    print(f"  Std:  {all_time_errors.std():.6f}")
    print(f"  Min:  {all_time_errors.min():.6f}")
    print(f"  Max:  {all_time_errors.max():.6f}")
    print(f"  Median: {np.median(all_time_errors):.6f}")
    
    print(f"\nFrequency Domain Reconstruction Error:")
    print(f"  Mean: {all_freq_errors.mean():.6f}")
    print(f"  Std:  {all_freq_errors.std():.6f}")
    print(f"  Min:  {all_freq_errors.min():.6f}")
    print(f"  Max:  {all_freq_errors.max():.6f}")
    print(f"  Median: {np.median(all_freq_errors):.6f}")
    
    print(f"\nKL Divergence:")
    print(f"  Mean: {all_kl_divs.mean():.6f}")
    print(f"  Std:  {all_kl_divs.std():.6f}")
    print(f"  Min:  {all_kl_divs.min():.6f}")
    print(f"  Max:  {all_kl_divs.max():.6f}")
    print("="*70)
    
    return {
        'time_errors': all_time_errors,
        'freq_errors': all_freq_errors,
        'kl_divs': all_kl_divs
    }


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading ST-VAE model...")
    model = ST_VAE(
        n_leads=12,
        seq_len=1000,
        latent_dim=32,
        beta=0.5,
        freq_weight=0.3,
        dropout=0.2
    ).to(device)
    
    weights_path = ROOT / 'src/weights/st_vae_weights/best_model.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Model loaded from {weights_path}")
    
    # Load datasets
    print("\nLoading datasets...")
    normal_dir = ROOT / 'data/processed/ptbxl'
    abnormal_dir = ROOT / 'data/processed/ptbxl_abnormal'
    
    normal_dataset = ECGDataset(normal_dir)
    abnormal_dataset = ECGDataset(abnormal_dir)
    
    print(f"Normal samples: {len(normal_dataset)}")
    print(f"Abnormal samples: {len(abnormal_dataset)}")
    
    # Create output directory
    output_dir = ROOT / 'outputs/st_vae/reconstructions'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test on normal ECGs
    print("\n" + "="*70)
    print("TESTING ON NORMAL ECGs")
    print("="*70)
    normal_error = plot_reconstruction_comparison(
        model, normal_dataset, device, num_samples=3,
        save_path=output_dir / 'normal_reconstruction.png'
    )
    print(f"Average reconstruction error (normal): {normal_error:.6f}")
    
    normal_stats = compute_reconstruction_statistics(model, normal_dataset, device, num_samples=100)
    
    # Test on abnormal ECGs
    print("\n" + "="*70)
    print("TESTING ON ABNORMAL ECGs")
    print("="*70)
    abnormal_error = plot_reconstruction_comparison(
        model, abnormal_dataset, device, num_samples=3,
        save_path=output_dir / 'abnormal_reconstruction.png'
    )
    print(f"Average reconstruction error (abnormal): {abnormal_error:.6f}")
    
    abnormal_stats = compute_reconstruction_statistics(model, abnormal_dataset, device, num_samples=100)
    
    # Compare normal vs abnormal
    print("\n" + "="*70)
    print("NORMAL vs ABNORMAL COMPARISON")
    print("="*70)
    print(f"Time domain error ratio (abnormal/normal): {abnormal_stats['time_errors'].mean() / normal_stats['time_errors'].mean():.3f}x")
    print(f"Frequency domain error ratio (abnormal/normal): {abnormal_stats['freq_errors'].mean() / normal_stats['freq_errors'].mean():.3f}x")
    print(f"KL divergence ratio (abnormal/normal): {abnormal_stats['kl_divs'].mean() / normal_stats['kl_divs'].mean():.3f}x")
    
    # Plot error distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Time domain errors
    axes[0].hist(normal_stats['time_errors'], bins=30, alpha=0.6, label='Normal', color='blue', density=True)
    axes[0].hist(abnormal_stats['time_errors'], bins=30, alpha=0.6, label='Abnormal', color='red', density=True)
    axes[0].set_xlabel('Time Domain Reconstruction Error')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Time Domain Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Frequency domain errors
    axes[1].hist(normal_stats['freq_errors'], bins=30, alpha=0.6, label='Normal', color='blue', density=True)
    axes[1].hist(abnormal_stats['freq_errors'], bins=30, alpha=0.6, label='Abnormal', color='red', density=True)
    axes[1].set_xlabel('Frequency Domain Reconstruction Error')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Frequency Domain Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL divergence
    axes[2].hist(normal_stats['kl_divs'], bins=30, alpha=0.6, label='Normal', color='blue', density=True)
    axes[2].hist(abnormal_stats['kl_divs'], bins=30, alpha=0.6, label='Abnormal', color='red', density=True)
    axes[2].set_xlabel('KL Divergence')
    axes[2].set_ylabel('Density')
    axes[2].set_title('KL Divergence Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved error distributions to {output_dir / 'error_distributions.png'}")
    plt.show()
    plt.close()
    
    print("\n" + "="*70)
    print("RECONSTRUCTION TEST COMPLETE")
    print("="*70)
    print(f"Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
