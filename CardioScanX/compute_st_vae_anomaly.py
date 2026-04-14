import torch
import numpy as np
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_st_vae_reconstruction_and_anomaly(model, sig_np, window_size=1000, stride=500):
    """
    Compute ST-VAE reconstruction and anomaly score using NLL + beta*KL (matches evaluation).
    
    Args:
        model: loaded ST-VAE pytorch model
        sig_np: np.ndarray [12, T] - ECG signal (12 leads, T samples)
        window_size: int - window size for sliding windows (default: 1000)
        stride: int - stride for sliding windows (default: 500)
    
    Returns:
        recon_mean: np.ndarray [12, T] - reconstructed signal mean
        recon_std: np.ndarray [12, T] - reconstructed signal std (from x_logvar)
        anomaly_per_timestep: np.ndarray [T, 12] - per-timestep anomaly map
        anomaly_score: float - overall anomaly score (NLL + beta*KL)
    """
    
    # Determine device from model
    device = next(model.parameters()).device
    
    # Convert to torch tensor
    sig = torch.tensor(sig_np, dtype=torch.float32)
    
    # Ensure shape is [12, T]
    if sig.ndim != 2:
        raise ValueError(f"Expected 2D signal, got {sig.ndim}D")
    if sig.shape[0] != 12:
        if sig.shape[1] == 12:
            sig = sig.T
        else:
            raise ValueError(f"Expected 12 leads, got shape {sig.shape}")
    
    T = sig.shape[1]
    
    # Pad if necessary
    if T < window_size:
        sig = F.pad(sig, (0, window_size - T))
        T = window_size
    
    # Slide windows: [12, T] -> [n_windows, window_size, 12]
    sig_T = sig.T  # [T, 12]
    windows = []
    window_positions = []
    for start in range(0, T - window_size + 1, stride):
        windows.append(sig_T[start:start+window_size])
        window_positions.append(start)
    
    if len(windows) == 0:
        # Signal too short, use entire signal
        windows = [sig_T[:window_size]]
        window_positions = [0]
    
    windows_tensor = torch.stack(windows).to(device)  # [n_windows, window_size, 12]
    
    # Forward pass through ST-VAE
    model.eval()
    with torch.no_grad():
        # ST-VAE returns: (x_mean, x_logvar, mu, logvar, ...)
        outputs = model(windows_tensor)
        x_mean = outputs[0]      # [n_windows, window_size, 12]
        x_logvar = outputs[1]    # [n_windows, window_size, 12]
        mu = outputs[2]          # [n_windows, window_size, latent_dim] - expanded
        logvar = outputs[3]      # [n_windows, window_size, latent_dim] - expanded
        
        # Extract compact mu/logvar (first timestep)
        mu_compact = mu[:, 0, :]      # [n_windows, latent_dim]
        logvar_compact = logvar[:, 0, :]  # [n_windows, latent_dim]
        
        # Extract beta from model
        beta = model.beta if hasattr(model, 'beta') else 0.5
        
        # Compute per-window metrics
        window_scores = []
        for i in range(len(windows)):
            # Variance-weighted reconstruction loss (NLL without constant)
            sigma2 = torch.exp(x_logvar[i])
            nll = (0.5 * torch.log(2 * np.pi * sigma2) + 
                   (windows_tensor[i] - x_mean[i])**2 / (2 * sigma2)).mean()
            
            # KL divergence for this window
            kl = -0.5 * (1 + logvar_compact[i] - mu_compact[i].pow(2) - logvar_compact[i].exp()).sum()
            
            # Combined score (matches evaluation.py)
            score = nll + beta * kl
            window_scores.append(score.item())
        
        # Overall anomaly score: mean across windows
        anomaly_score = float(np.mean(window_scores))
        
        # Reconstruct full signal by averaging overlapping windows
        recon_mean_full = torch.zeros((T, 12), device=device)
        recon_var_full = torch.zeros((T, 12), device=device)
        counts = torch.zeros((T, 12), device=device)
        
        for i, start in enumerate(window_positions):
            end = start + window_size
            recon_mean_full[start:end] += x_mean[i]
            recon_var_full[start:end] += torch.exp(x_logvar[i])
            counts[start:end] += 1
        
        # Average overlapping regions
        counts[counts == 0] = 1
        recon_mean_full = recon_mean_full / counts
        recon_var_full = recon_var_full / counts
        recon_std_full = torch.sqrt(recon_var_full)
        
        # recon_mean_full and recon_var_full are [T, 12]
        # Convert to numpy [12, T] for output
        recon_mean_np = recon_mean_full.T.cpu().numpy()  # [12, T]
        recon_std_np = recon_std_full.T.cpu().numpy()    # [12, T]
        
        # Compute per-timestep anomaly map [T, 12]
        # Use variance-weighted squared error
        # sig was already normalized to [12, T] above, use it instead of sig_np
        orig_signal = sig[:, :T].T.cpu().numpy()  # [T, 12]
        recon_mean_for_error = recon_mean_full.cpu().numpy()  # [T, 12]
        recon_var_for_error = recon_var_full.cpu().numpy()    # [T, 12]
        variance_weighted_error = (orig_signal - recon_mean_for_error)**2 / (recon_var_for_error + 1e-6)
        
        return recon_mean_np, recon_std_np, variance_weighted_error, anomaly_score


def normalize_score(raw_score, min_score=45.0, max_score=85.0):
    """
    Normalize a raw anomaly score to [0, 1] range for presentation.
    
    Args:
        raw_score: float - raw anomaly score (typically 45-85 range)
        min_score: float - minimum observed score in dataset
        max_score: float - maximum observed score in dataset
    
    Returns:
        float - normalized score in [0, 1] range
    """
    normalized = (raw_score - min_score) / (max_score - min_score)
    return np.clip(normalized, 0.0, 1.0)


def compute_st_vae_anomaly_simple(model, sig_np, normalized=True):
    """
    Simplified version that returns just the anomaly score.
    
    Args:
        model: loaded ST-VAE pytorch model
        sig_np: np.ndarray [12, T] - ECG signal
        normalized: bool - if True, return score normalized to [0, 1] range (default: True)
    
    Returns:
        anomaly_score: float - NLL + beta*KL anomaly score
                               Raw range: ~28k-82k, Normalized range: [0, 1]
    """
    _, _, _, score = compute_st_vae_reconstruction_and_anomaly(model, sig_np)
    
    if normalized:
        # Normalize based on empirically observed min/max from full dataset
        # Normal range: ~50-55, Abnormal range: ~75-80
        min_score = 45.0  # Observed minimum across dataset
        max_score = 85.0  # Observed maximum across dataset
        normalized_score = (score - min_score) / (max_score - min_score)
        normalized_score = np.clip(normalized_score, 0.0, 1.0)  # Ensure [0, 1]
        return normalized_score
    
    return score


def compute_st_vae_metrics_breakdown(model, sig_np):
    """
    Get detailed breakdown of ST-VAE anomaly detection metrics.
    
    Args:
        model: loaded ST-VAE pytorch model
        sig_np: np.ndarray [12, T] - ECG signal
    
    Returns:
        dict with keys:
            - 'mse': simple MSE reconstruction error
            - 'weighted_mse': variance-weighted MSE
            - 'kl_divergence': KL divergence of latent space
            - 'nll': negative log-likelihood
            - 'total_score': NLL + beta*KL (primary metric)
            - 'beta': beta value used
    """
    # Determine device from model
    device = next(model.parameters()).device
    
    sig = torch.tensor(sig_np, dtype=torch.float32)
    
    # Ensure shape [12, T]
    if sig.shape[0] != 12:
        sig = sig.T
    
    # Take first 1000 samples for quick evaluation
    sig_window = sig[:, :1000].T.unsqueeze(0).to(device)  # [1, 1000, 12]
    
    model.eval()
    with torch.no_grad():
        outputs = model(sig_window)
        x_mean = outputs[0]
        x_logvar = outputs[1]
        mu = outputs[2]
        logvar = outputs[3]
        
        beta = model.beta if hasattr(model, 'beta') else 0.5
        
        # 1. Simple MSE
        mse = F.mse_loss(x_mean, sig_window).item()
        
        # 2. Variance-weighted MSE
        sigma2 = torch.exp(x_logvar)
        weighted_mse = ((sig_window - x_mean)**2 / (sigma2 + 1e-6)).mean().item()
        
        # 3. KL divergence
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum().item()
        
        # 4. NLL (negative log-likelihood)
        nll = (0.5 * torch.log(2 * np.pi * sigma2) + 
               (sig_window - x_mean)**2 / (2 * sigma2)).mean().item()
        
        # 5. Total score
        total_score = nll + beta * kl
        
        return {
            'mse': mse,
            'weighted_mse': weighted_mse,
            'kl_divergence': kl,
            'nll': nll,
            'total_score': total_score,
            'beta': beta
        }


# Example usage:
if __name__ == "__main__":
    print("ST-VAE Anomaly Detection Module")
    print("="*70)
    print("\nThis module provides three main functions:")
    print("1. compute_st_vae_reconstruction_and_anomaly() - Full reconstruction and anomaly detection")
    print("2. compute_st_vae_anomaly_simple() - Quick anomaly score only")
    print("3. compute_st_vae_metrics_breakdown() - Detailed metrics breakdown")
    print("\nUsage:")
    print("  from compute_st_vae_anomaly import compute_st_vae_reconstruction_and_anomaly")
    print("  recon_mean, recon_std, anomaly_map, score = compute_st_vae_reconstruction_and_anomaly(model, ecg_signal)")
    print("="*70)
