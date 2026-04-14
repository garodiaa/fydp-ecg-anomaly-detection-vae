import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_reconstructions import *
import numpy as np
import torch
import torch.nn.functional as F

def test_model_anomaly_detection(model_name):
    """Test anomaly detection for different model architectures"""
    
    model = load_model(model_name)
    device = next(model.parameters()).device
    
    # Get samples
    normal_idx = 100
    abnormal_idx = 15000
    ecg_normal, _ = get_sample_ecg(normal_idx)
    ecg_abnormal, _ = get_sample_ecg(abnormal_idx)
    
    x_normal = ecg_normal.unsqueeze(0).to(device)
    x_abnormal = ecg_abnormal.unsqueeze(0).to(device)
    
    print(f'\n{"="*70}')
    print(f'{model_name.upper()} Anomaly Detection Analysis')
    print(f'{"="*70}')
    
    with torch.no_grad():
        out_n = model(x_normal)
        out_a = model(x_abnormal)
        
        # ---------------------------------------------------------
        # 1. CAE - Deterministic (NO VAE components)
        # ---------------------------------------------------------
        if 'CAE' in model_name.upper() or model.__class__.__name__ == 'CAE':
            # CAE only outputs reconstruction, no mu/logvar
            x_recon_n = out_n  # [batch, 12, seq_len]
            x_recon_a = out_a
            
            mse_n = F.mse_loss(x_recon_n, x_normal).item()
            mse_a = F.mse_loss(x_recon_a, x_abnormal).item()
            
            print(f'\n📊 Reconstruction MSE (ONLY metric for CAE):')
            print(f'  Normal ECG:   {mse_n:.4f}')
            print(f'  Abnormal ECG: {mse_a:.4f}')
            print(f'  Ratio: {mse_a/mse_n:.2f}x')
            
            if mse_a > mse_n * 1.5:
                print(f'\n✓ CAE can detect anomalies!')
                print(f'  Expected F1 ≈ 0.30-0.45 (baseline)')
            else:
                print(f'\n✗ CAE struggles - deterministic models have limitations')
                
        # ---------------------------------------------------------
        # 2. Standard VAEs with Homoscedastic Noise
        # (BA-VAE, HL-VAE, VAE-GRU, VAE-BiLSTM-Attn)
        # ---------------------------------------------------------
        elif model_name in ['ba_vae', 'hlvae', 'vae_gru', 'vae_bilstm_attn']:
            # These models output: (x_mean, x_logvar, mu, logvar, attn_weights, lead_w)
            # mu and logvar are (B, T, latent_dim) - use first timestep
            x_mean_n, x_logvar_n, mu_n, logvar_n = out_n[0], out_n[1], out_n[2], out_n[3]
            x_mean_a, x_logvar_a, mu_a, logvar_a = out_a[0], out_a[1], out_a[2], out_a[3]
            
            # mu/logvar are expanded (B, T, latent_dim), take first timestep
            mu_n = mu_n[:, 0, :]  # (B, latent_dim)
            mu_a = mu_a[:, 0, :]
            logvar_n = logvar_n[:, 0, :]
            logvar_a = logvar_a[:, 0, :]
            
            # MSE
            mse_n = F.mse_loss(x_mean_n, x_normal).item()
            mse_a = F.mse_loss(x_mean_a, x_abnormal).item()
            
            # KL divergence
            kl_n = (-0.5 * (1 + logvar_n - mu_n.pow(2) - logvar_n.exp())).sum().item()
            kl_a = (-0.5 * (1 + logvar_a - mu_a.pow(2) - logvar_a.exp())).sum().item()
            
            # Combined (like training loss)
            beta = 0.1 if 'ba' in model_name else (0.01 if 'hl' in model_name else 0.001)
            score_n = mse_n + beta * kl_n
            score_a = mse_a + beta * kl_a
            
            print(f'\n[METRICS] {model_name.upper()} (Homoscedastic):')
            print(f'\n1. Reconstruction MSE:')
            print(f'  Normal:   {mse_n:.4f}')
            print(f'  Abnormal: {mse_a:.4f}')
            print(f'  Ratio: {mse_a/mse_n:.2f}x')
            
            print(f'\n2. KL Divergence (β={beta}):')
            print(f'  Normal:   {kl_n:.2f}')
            print(f'  Abnormal: {kl_a:.2f}')
            print(f'  Ratio: {kl_a/kl_n:.2f}x')
            
            print(f'\n3. Combined Score (MSE + β·KL):')
            print(f'  Normal:   {score_n:.4f}')
            print(f'  Abnormal: {score_a:.4f}')
            print(f'  Ratio: {score_a/score_n:.2f}x')
            
            if score_a > score_n * 1.3:
                print(f'\n✓ {model_name.upper()} can detect anomalies!')
                if 'ba' in model_name:
                    print(f'  Expected F1 ~ 0.55, Precision=0.91 (low false alarms)')
                elif 'hl' in model_name:
                    print(f'  Expected F1 ~ 0.805 (hierarchical features)')
                elif 'gru' in model_name:
                    print(f'  Expected F1 ~ 0.77 (fast sequential)')
                elif 'attention' in model_name or 'attn' in model_name:
                    print(f'  Expected F1 ~ 0.785 (attention-based)')
                        
        # ---------------------------------------------------------
        # 3. Advanced VAEs with Heteroscedastic Noise
        # (ST-VAE, VAE-BiLSTM-MHA)
        # ---------------------------------------------------------
        elif model_name in ['st_vae', 'vae_bilstm_mha']:
            # These output: (x_mean, x_logvar, mu, logvar) OR more items
            # Handle variable-length outputs
            if len(out_n) >= 4:
                x_mean_n, x_logvar_n, mu_n, logvar_n = out_n[0], out_n[1], out_n[2], out_n[3]
                x_mean_a, x_logvar_a, mu_a, logvar_a = out_a[0], out_a[1], out_a[2], out_a[3]
            else:
                print(f'  ⚠️ Unexpected output format: {len(out_n)} outputs')
                return
            
            # Simple MSE
            mse_n = F.mse_loss(x_mean_n, x_normal).item()
            mse_a = F.mse_loss(x_mean_a, x_abnormal).item()
            
            # KL divergence
            kl_n = (-0.5 * (1 + logvar_n - mu_n.pow(2) - logvar_n.exp())).sum().item()
            kl_a = (-0.5 * (1 + logvar_a - mu_a.pow(2) - logvar_a.exp())).sum().item()
            
            # Variance-weighted MSE (KEY for heteroscedastic models!)
            sigma2_n = torch.exp(x_logvar_n)
            sigma2_a = torch.exp(x_logvar_a)
            weighted_mse_n = ((x_normal - x_mean_n)**2 / (sigma2_n + 1e-6)).mean().item()
            weighted_mse_a = ((x_abnormal - x_mean_a)**2 / (sigma2_a + 1e-6)).mean().item()
            
            # NLL loss (what's actually used in training!)
            nll_n = (0.5 * torch.log(2 * np.pi * sigma2_n) + 
                    (x_normal - x_mean_n)**2 / (2 * sigma2_n)).mean().item()
            nll_a = (0.5 * torch.log(2 * np.pi * sigma2_a) + 
                    (x_abnormal - x_mean_a)**2 / (2 * sigma2_a)).mean().item()
            
            beta = 0.1 if 'st' in model_name else 0.01
            score_n = nll_n + beta * kl_n
            score_a = nll_a + beta * kl_a
            
            print(f'\n[METRICS] {model_name.upper()} (Heteroscedastic):')
            print(f'\n1. Simple MSE (less important):')
            print(f'  Normal:   {mse_n:.4f}')
            print(f'  Abnormal: {mse_a:.4f}')
            print(f'  Ratio: {mse_a/mse_n:.2f}x')
            
            print(f'\n2. Weighted MSE (accounts for uncertainty):')
            print(f'  Normal:   {weighted_mse_n:.4f}')
            print(f'  Abnormal: {weighted_mse_a:.4f}')
            print(f'  Ratio: {weighted_mse_a/weighted_mse_n:.2f}x')
            
            print(f'\n3. KL Divergence (β={beta}):')
            print(f'  Normal:   {kl_n:.2f}')
            print(f'  Abnormal: {kl_a:.2f}')
            print(f'  Ratio: {kl_a/kl_n:.2f}x')
            
            print(f'\n4. NLL + β·KL (TRUE EVALUATION METRIC):')
            print(f'  Normal:   {score_n:.4f}')
            print(f'  Abnormal: {score_a:.4f}')
            print(f'  Ratio: {score_a/score_n:.2f}x {"← PRIMARY METRIC" if score_a > score_n * 1.2 else ""}')
            
            if score_a > score_n * 1.2:
                print(f'\n✓ {model_name.upper()} can detect anomalies!')
                if 'st' in model_name:
                    print(f'  Expected F1 ~ 0.90, AUC=0.93 [BEST MODEL]')
                    print(f'  Dual time-frequency + uncertainty = powerful!')
                elif 'mha' in model_name:
                    print(f'  Expected F1 ~ 0.825 [2ND BEST]')
                    print(f'  Multi-head attention captures complex patterns')
                    
        else:
            print(f'⚠️ Unknown model type: {model_name}')
            
    print(f'{"="*70}\n')


# Test all models (use correct model names from generate_reconstructions.py)
models_to_test = [
    # 'cae',              # Not available in load_model()
    'ba_vae',          # Homoscedastic VAE
    'hlvae',           # Homoscedastic hierarchical (note: hlvae not hl_vae)
    'vae_gru',         # Homoscedastic sequential
    'vae_bilstm_attn',  # Homoscedastic with attention (note: attn not attention)
    'st_vae',          # Heteroscedastic (best)
    'vae_bilstm_mha',  # Heteroscedastic with MHA
]

for model_name in models_to_test:
    try:
        test_model_anomaly_detection(model_name)
    except Exception as e:
        print(f'\n[ERROR] Error testing {model_name}: {e}\n')
        import traceback
        traceback.print_exc()