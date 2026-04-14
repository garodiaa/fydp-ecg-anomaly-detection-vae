import numpy as np
import torch
from pathlib import Path
import sys
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
# import original VAE helpers; safe to call functions from src instead of duplicating big code
try:
    from src.visualization_vae import slide_windows as vae_slide_windows
    from src.visualization_vae import reconstruct_full_mean_std as vae_reconstruct_full_mean_std
    from src.visualization_vae import ecg_score as vae_ecg_score
except Exception:
    vae_slide_windows = None
    vae_reconstruct_full_mean_std = None
    vae_ecg_score = None

if vae_reconstruct_full_mean_std is None:
    # Implement a local fallback for VAE if original helper is not importable

    WINDOW = 500
    STRIDE = 250
    ALPHA = 0.7
    BETA = 0.3

    def vae_slide_windows_local(sig_t: torch.Tensor):
        if not isinstance(sig_t, torch.Tensor):
            sig_t = torch.tensor(sig_t, dtype=torch.float32)
        if sig_t.ndim != 2:
            raise ValueError(f"Expected 2D signal, got {sig_t.ndim}D")
        if sig_t.shape[0] == 12:
            sig_arr = sig_t
        elif sig_t.shape[1] == 12:
            sig_arr = sig_t.T
        else:
            if sig_t.shape[0] > 12:
                sig_arr = sig_t[:12, :]
            elif sig_t.shape[1] > 12:
                sig_arr = sig_t[:, :12].T
            else:
                raise ValueError('Unexpected signal shape')
        T = sig_arr.shape[1]
        if T < WINDOW:
            sig_arr = torch.nn.functional.pad(sig_arr, (0, WINDOW - T))
            T = WINDOW
        sig_T = sig_arr.T
        wins = []
        for start in range(0, T - WINDOW + 1, STRIDE):
            wins.append(sig_T[start:start+WINDOW])
        return torch.stack(wins)

    def vae_reconstruct_full_mean_std_local(model, sig_t: torch.Tensor):
        win_tensor = vae_slide_windows_local(sig_t).to(DEVICE)
        with torch.no_grad():
            x_mean, x_logvar, *_ = model.forward(win_tensor)
        x_std = torch.exp(0.5 * x_logvar)
        recon_m = torch.zeros((5000, 12), device=DEVICE)
        recon_s = torch.zeros((5000, 12), device=DEVICE)
        counts = torch.zeros((5000, 12), device=DEVICE)
        for i, start in enumerate(range(0, 5000 - WINDOW + 1, STRIDE)):
            recon_m[start:start+WINDOW] += x_mean[i]
            recon_s[start:start+WINDOW] += x_std[i]
            counts[start:start+WINDOW] += 1
        counts[counts == 0] = 1
        recon_m = (recon_m / counts).T.cpu()
        recon_s = (recon_s / counts).T.cpu()
        return recon_m, recon_s

    def vae_ecg_score_local(model, sig_t: torch.Tensor):
        win_tensor = vae_slide_windows_local(sig_t).to(DEVICE)
        with torch.no_grad():
            outs = model.forward(win_tensor)
        # Unpack outputs depending on the model
        if isinstance(outs, tuple) or isinstance(outs, list):
            n = len(outs)
        else:
            # Some models might return a single tensor
            outs = (outs,)
            n = 1

        # Default values
        x_mean = None
        x_logvar = None
        mu = None
        logvar = None
        attn = None

        if n == 6:
            x_mean, x_logvar, mu, logvar, attn, _ = outs
        elif n == 7:
            # VAE_MHA returns (x_mean, x_logvar, x_recon, z_mean, z_logvar, attn, focus)
            x_mean, x_logvar, _x_recon, z_mean, z_logvar, attn, _ = outs
            mu, logvar = z_mean, z_logvar
        elif n == 4:
            # GRU VAE returns (x_recon, x_logvar, mu, logvar)
            x_mean, x_logvar, mu, logvar = outs
            attn = None
        elif n == 2:
            # Some minimal models return only mean and logvar
            x_mean, x_logvar = outs
            mu = torch.zeros_like(x_logvar)
            logvar = torch.zeros_like(x_logvar)
            attn = None
        else:
            # Fallback: attempt best-effort unpacking
            try:
                x_mean, x_logvar = outs[0], outs[1]
            except Exception:
                raise RuntimeError('Unrecognized VAE forward output signature: ' + str(outs))
        sigma2 = torch.exp(x_logvar)
        err = (win_tensor - x_mean) ** 2 / (sigma2 + 1e-6)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=[1, 2])
        var_term = x_logvar
        base = ALPHA * err.sum(dim=[1, 2]) + (1 - ALPHA) * var_term.sum(dim=[1, 2]) + BETA * kl
        if attn is None:
            # no attention available, create a zeros placeholder with shape (B, T, T)
            B, T, _ = x_mean.shape
            attn = torch.zeros((B, T, T), device=x_mean.device)
        if attn.dim() == 4:
            alpha = attn.mean(dim=(1, 2, 3))
        else:
            alpha = attn.mean(dim=(1, 2))
        scores = base
        return float(scores.mean())

try:
    from src.visualization_cae import slide_windows as cae_slide_windows
    from src.visualization_cae import reconstruct_full_mean_std as cae_reconstruct_full_mean_std
    from src.visualization_cae import ecg_score as cae_ecg_score
except Exception:
    cae_slide_windows = None
    cae_reconstruct_full_mean_std = None
    cae_ecg_score = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_reconstruction_and_anomaly(model, sig_np, model_name):
    """
    model: loaded pytorch model
    sig_np: np.ndarray [12, T]
    model_name: one of 'CAE', 'VAE', 'VAE_MHA', 'VAE_GRU'
    return: recon (12,T), std (12,T), anomaly_per_lead (T,12), score(float)
    """
    # Convert np to torch for the visualization helpers
    sig = torch.tensor(sig_np, dtype=torch.float32)

    if 'CA' in model.__class__.__name__ or model_name.startswith('CAE'):
        # CAE uses CAE wrapper
        if cae_reconstruct_full_mean_std is None:
            # fallback: implement basic CAE reconstruction and score similar to original helpers

            WINDOW = 500
            STRIDE = 250

            def cae_slide_windows_local(sig_t: torch.Tensor):
                if not isinstance(sig_t, torch.Tensor):
                    sig_t = torch.tensor(sig_t, dtype=torch.float32)
                if sig_t.ndim != 2:
                    raise ValueError(f"Expected 2D signal, got {sig_t.ndim}D")
                if sig_t.shape[0] == 12:
                    sig_arr = sig_t
                elif sig_t.shape[1] == 12:
                    sig_arr = sig_t.T
                else:
                    if sig_t.shape[0] > 12:
                        sig_arr = sig_t[:12, :]
                    elif sig_t.shape[1] > 12:
                        sig_arr = sig_t[:, :12].T
                    else:
                        raise ValueError("Unexpected signal shape")
                T = sig_arr.shape[1]
                if T < WINDOW:
                    sig_arr = torch.nn.functional.pad(sig_arr, (0, WINDOW - T))
                    T = WINDOW
                sig_T = sig_arr.T
                wins = []
                for start in range(0, T - WINDOW + 1, STRIDE):
                    wins.append(sig_T[start:start+WINDOW])
                return torch.stack(wins)

            def cae_reconstruct_full_mean_std_local(model, sig_t: torch.Tensor):
                win_tensor = cae_slide_windows_local(sig_t).to(DEVICE)   # [n_windows, WINDOW, 12]
                win_in = win_tensor.permute(0, 2, 1)  # [n_windows, 12, WINDOW]
                with torch.no_grad():
                    recon_out = model(win_in)
                recon = recon_out.permute(0, 2, 1)
                recon_m = torch.zeros((5000, 12), device=DEVICE)
                recon_s = torch.zeros_like(recon_m)
                counts = torch.zeros_like(recon_m)
                for i, start in enumerate(range(0, 5000 - WINDOW + 1, STRIDE)):
                    recon_m[start:start+WINDOW] += recon[i]
                    counts[start:start+WINDOW] += 1
                counts[counts == 0] = 1
                recon_m = (recon_m / counts).T.cpu()
                recon_s = recon_s.T.cpu()
                return recon_m, recon_s

            def cae_ecg_score_local(model, sig_t: torch.Tensor):
                win_tensor = cae_slide_windows_local(sig_t).to(DEVICE)
                win_in = win_tensor.permute(0, 2, 1)
                with torch.no_grad():
                    recon_out = model(win_in)
                recon = recon_out.permute(0, 2, 1)
                err = (win_tensor - recon) ** 2
                return float(err.mean())

            recon, std = cae_reconstruct_full_mean_std_local(model, sig)
            score = cae_ecg_score_local(model, sig)
        # compute per-lead anomaly timeline: mse per sample
        # recon shape [12,5000], convert to [5000,12]
        recon_np = recon.numpy()
        std_np = std.numpy()
        orig = sig_np[:, :recon_np.shape[1]].T
        mse = (orig - recon_np.T) ** 2
        return recon_np, std_np, mse, float(score)

    else:
        # assume VAE
        if vae_reconstruct_full_mean_std is None:
            # fallback
            recon_m, recon_s = vae_reconstruct_full_mean_std_local(model, sig)
            score = vae_ecg_score_local(model, sig)
        else:
            recon_m, recon_s = vae_reconstruct_full_mean_std(model, sig)
            score = vae_ecg_score(model, sig)
        recon_np = recon_m.numpy()
        std_np = recon_s.numpy()
        orig = sig_np[:, :recon_np.shape[1]].T
        mse = (orig - recon_np.T) ** 2
        return recon_np, std_np, mse, float(score)
