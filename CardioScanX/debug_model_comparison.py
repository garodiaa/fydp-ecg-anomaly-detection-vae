"""
Debug: Compare model outputs between streamlit loader and original loader
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Test with streamlit loader
print("=" * 80)
print("TEST 1: Using streamlit model_loader")
print("=" * 80)
from model_loader import load_model_by_name

model1, _ = load_model_by_name('ST-VAE (Spatiotemporal VAE)', map_location='cpu')
model1.eval()
print(f"Model 1 Beta: {model1.beta}")
print(f"Model 1 Device: {next(model1.parameters()).device}")

# Load sample
ecg_data = np.load('SampleData/normal/00011_hr.npy')  # (12, 5000)
if ecg_data.shape[1] < 5000:
    ecg_data = np.pad(ecg_data, ((0,0),(0, 5000 - ecg_data.shape[1])), mode='constant')
else:
    ecg_data = ecg_data[:, :5000]

# Forward pass
sig = torch.tensor(ecg_data, dtype=torch.float32)
sig_window = sig[:, :1000].T.unsqueeze(0)  # [1, 1000, 12]

with torch.no_grad():
    outputs1 = model1(sig_window)
    mu1 = outputs1[2][0, :, :]  # [1000, latent_dim]
    logvar1 = outputs1[3][0, :, :]  # [1000, latent_dim]
    
    # Use first timestep
    mu_compact1 = mu1[0, :]
    logvar_compact1 = logvar1[0, :]
    
    kl1 = -0.5 * (1 + logvar_compact1 - mu_compact1.pow(2) - logvar_compact1.exp()).sum().item()
    
    print(f"\nOutput shapes:")
    print(f"  mu: {outputs1[2].shape}")
    print(f"  logvar: {outputs1[3].shape}")
    print(f"  mu_compact (first timestep): {mu_compact1.shape}")
    print(f"  KL (using first timestep): {kl1:.2f}")
    print(f"  mu_compact mean: {mu_compact1.mean().item():.4f}")
    print(f"  mu_compact std: {mu_compact1.std().item():.4f}")

# Test with original loader
print("\n" + "=" * 80)
print("TEST 2: Using original generate_reconstructions")
print("=" * 80)

# Add original to path
sys.path.insert(0, str(Path.cwd() / 'original'))
from generate_reconstructions import load_model as load_model_original

model2 = load_model_original('st_vae')
model2.eval()
print(f"Model 2 Beta: {model2.beta}")
print(f"Model 2 Device: {next(model2.parameters()).device}")

# Move to CPU for fair comparison
model2 = model2.cpu()

with torch.no_grad():
    outputs2 = model2(sig_window)
    mu2 = outputs2[2][0, :, :]  # [1000, latent_dim]
    logvar2 = outputs2[3][0, :, :]  # [1000, latent_dim]
    
    # Use first timestep
    mu_compact2 = mu2[0, :]
    logvar_compact2 = logvar2[0, :]
    
    kl2 = -0.5 * (1 + logvar_compact2 - mu_compact2.pow(2) - logvar_compact2.exp()).sum().item()
    
    print(f"\nOutput shapes:")
    print(f"  mu: {outputs2[2].shape}")
    print(f"  logvar: {outputs2[3].shape}")
    print(f"  mu_compact (first timestep): {mu_compact2.shape}")
    print(f"  KL (using first timestep): {kl2:.2f}")
    print(f"  mu_compact mean: {mu_compact2.mean().item():.4f}")
    print(f"  mu_compact std: {mu_compact2.std().item():.4f}")

print("\n" + "=" * 80)
print("COMPARISON:")
print("=" * 80)
print(f"KL difference: {abs(kl1 - kl2):.6f}")
print(f"mu difference (L2): {torch.norm(mu_compact1 - mu_compact2).item():.6f}")
print(f"Are models identical? {torch.allclose(mu_compact1, mu_compact2, atol=1e-5)}")
