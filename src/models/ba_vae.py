# src/models/beat_vae.py
"""
Beat-Aligned VAE (BA-VAE)

- Designed to be trained on beat-centered windows extracted from 12-lead ECGs.
- Input shape for model: (B, T, n_leads)  where T is beat length (e.g., 200-400)
- Encoder: small GRU -> per-timestep mu/logvar or pooled mu/logvar (we do per-beat global latent here)
- Decoder: map latent -> per-timepoint reconstruction (reshape)
- Loss: Negative log-likelihood under Normal(x_mean, exp(0.5*x_logvar)) + beta * KL

Utilities:
- extract_beats(signal, fs=500, pre_r=0.2, post_r=0.4): extract beat windows around R-peaks (seconds)
- BeatDataset: dataset yielding beat windows (lazy extraction from .npy files)

Author: ChatGPT (tailored to your repo)
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.data import Dataset
from scipy.signal import find_peaks


class BeatVAE(nn.Module):
    def __init__(
        self,
        n_leads: int = 12,
        beat_len: int = 400,
        enc_hidden: int = 128,
        n_layers: int = 1,
        latent_dim: int = 16,
        beta: float = 0.01,
        dropout: float = 0.05,
    ):
        """
        Args:
            n_leads: number of ECG leads (12)
            beat_len: number of time samples per beat window (T)
            enc_hidden: GRU hidden size
            n_layers: GRU layers
            latent_dim: latent dimensionality per-beat
            beta: KL weight
        """
        super().__init__()
        self.n_leads = n_leads
        self.beat_len = beat_len
        self.enc_hidden = enc_hidden
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.beta = beta

        # Bidirectional GRU encoder: processes (B, T, n_leads) -> h_seq (B, T, H*2)
        self.gru_enc = nn.GRU(
            input_size=n_leads, 
            hidden_size=enc_hidden, 
            num_layers=n_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Layer norm for stability
        self.ln_enc = nn.LayerNorm(enc_hidden * 2)

        # Map final hidden -> stats (bidirectional = 2x hidden)
        self.to_stats = nn.Sequential(
            nn.Linear(enc_hidden * 2, enc_hidden),
            nn.LayerNorm(enc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden, 2 * latent_dim),
        )

        # Decoder: map latent -> sequence
        self.latent_to_init = nn.Linear(latent_dim, enc_hidden)
        self.gru_dec = nn.GRU(
            input_size=latent_dim, 
            hidden_size=enc_hidden, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=False
        )
        
        self.ln_dec = nn.LayerNorm(enc_hidden)
        
        # Improved decoder output
        self.to_recon = nn.Sequential(
            nn.Linear(enc_hidden, enc_hidden),
            nn.LayerNorm(enc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden, 2 * n_leads),  # mean & logvar per lead
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        """
        x: (B, T, n_leads)
        returns: mu (B, latent_dim), logvar (B, latent_dim)
        """
        h_seq, h_n = self.gru_enc(x)  # h_seq: (B, T, H*2) for bidirectional
        # Apply layer norm
        h_seq = self.ln_enc(h_seq)
        # Use last timestep (combines forward and backward)
        h_last = h_seq[:, -1, :]  # (B, H*2)
        stats = self.to_stats(h_last)  # (B, 2*latent_dim)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        z: (B, latent_dim)
        returns x_mean (B, T, n_leads), x_logvar (B, T, n_leads)
        """
        B = z.size(0)
        # Feed latent as repeated inputs across timesteps to GRU decoder
        z_in = z.unsqueeze(1).expand(-1, self.beat_len, -1)  # (B, T, latent_dim)
        h0 = torch.tanh(self.latent_to_init(z)).unsqueeze(0)  # (1, B, H) with tanh activation
        d_seq, _ = self.gru_dec(z_in, h0)  # (B, T, H)
        
        # Apply layer norm
        d_seq = self.ln_dec(d_seq)
        
        # Reshape and reconstruct
        rec_stats = self.to_recon(d_seq.reshape(B * self.beat_len, -1))  # (B*T, 2*n_leads)
        x_mean, x_logvar = rec_stats.chunk(2, dim=-1)
        x_mean = x_mean.reshape(B, self.beat_len, self.n_leads)
        x_logvar = x_logvar.reshape(B, self.beat_len, self.n_leads)
        return x_mean, x_logvar

    def forward(self, x):
        """
        x: (B, T, n_leads)
        returns: x_mean, x_logvar, mu, logvar, attn_weights, lead_w (6 values for compatibility)
        """
        B, T, C = x.shape
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_mean, x_logvar = self.decode(z)
        
        # Expand mu/logvar to (B, T, latent_dim) for compatibility with diagnostics
        mu_expanded = mu.unsqueeze(1).expand(-1, T, -1)
        logvar_expanded = logvar.unsqueeze(1).expand(-1, T, -1)
        
        # Placeholder attention weights (uniform) for compatibility
        attn_weights = torch.ones(B, T, T, device=x.device) / T
        lead_w = torch.ones(B, T, C, device=x.device) / C
        
        return x_mean, x_logvar, mu_expanded, logvar_expanded, attn_weights, lead_w

    def loss_function(self, x, x_mean, x_logvar, mu, logvar, free_bits=0.0):
        """
        Negative log-likelihood under Normal(x_mean, exp(0.5*x_logvar)) + beta * KL
        Args:
            x: input (B, T, n_leads)
            x_mean: reconstruction mean (B, T, n_leads)
            x_logvar: reconstruction log variance (B, T, n_leads)
            mu: latent mean (B, T, latent_dim) - will use first timestep
            logvar: latent log variance (B, T, latent_dim) - will use first timestep
            free_bits: minimum KL per dimension (helps prevent posterior collapse)
        Returns: total, recon_loss, kl_loss
        """
        # Use first timestep of mu/logvar (they're expanded, so all same)
        mu = mu[:, 0, :]  # (B, latent_dim)
        logvar = logvar[:, 0, :]  # (B, latent_dim)
        
        # Clamp for numerical stability
        x_logvar = torch.clamp(x_logvar, min=-10.0, max=10.0)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        # Reconstruction loss (negative log-likelihood)
        dist = Normal(x_mean, torch.exp(0.5 * x_logvar) + 1e-6)
        log_px = dist.log_prob(x)  # shape (B, T, n_leads)
        recon_loss = -log_px.sum(dim=[1, 2]).mean()  # sum over time & leads, mean over batch

        # KL divergence with free bits
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent_dim)
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        kl_loss = kl_per_dim.sum(dim=-1).mean()  # sum over latent dims, mean over batch

        total = recon_loss + self.beta * kl_loss
        return total, recon_loss, kl_loss


# --------------------------
# Utilities: beat extraction & dataset
# --------------------------
def extract_beats(signal: np.ndarray, fs: int = 500, pre_r: float = 0.2, post_r: float = 0.4, beat_len: int = 400):
    """
    Extract beat windows around R-peaks.

    Args:
        signal: np.ndarray shape (12, T) or (T, 12)
        fs: sampling frequency (Hz)
        pre_r: seconds before R to include
        post_r: seconds after R to include
        beat_len: desired output length (samples). If pre_r+post_r doesn't match beat_len, we will center/pad/truncate.
    Returns:
        beats: np.ndarray shape (n_beats, beat_len, 12)
    """
    # Normalize orientation: want shape (T, 12)
    if signal.ndim == 2 and signal.shape[0] == 12 and signal.shape[1] > 12:
        # probably (12, T) -> transpose
        sig = signal.T
    elif signal.ndim == 2 and signal.shape[1] == 12 and signal.shape[0] != 12:
        sig = signal
    elif signal.ndim == 2 and signal.shape[0] == 12 and signal.shape[1] == 12:
        # ambiguous small, treat as (12,12) -> transpose to (12,12)
        sig = signal.T
    else:
        sig = signal

    T, C = sig.shape
    if C != 12:
        # try transpose fallback
        if signal.shape[0] == 12:
            sig = signal.T
            T, C = sig.shape

    # Use Lead II (index 1) for R-peak detection
    lead2 = sig[:, 1]
    # simple peak find: height and distance tuned to fs
    distance = int(0.3 * fs)  # at least 300ms apart
    # we use prominence to avoid noisy peaks; user can tweak
    peaks, _ = find_peaks(lead2, distance=distance, prominence=(None, None))

    half_pre = int(round(pre_r * fs))
    half_post = int(round(post_r * fs))
    desired_len = beat_len

    beats = []
    for p in peaks:
        start = p - half_pre
        end = p + half_post
        if start < 0 or end > T:
            continue
        win = sig[start:end, :]  # shape (window_len, 12)
        # pad/truncate to desired_len
        if win.shape[0] < desired_len:
            pad_top = (desired_len - win.shape[0]) // 2
            pad_bottom = desired_len - win.shape[0] - pad_top
            win = np.pad(win, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0.0)
        elif win.shape[0] > desired_len:
            # center crop
            excess = win.shape[0] - desired_len
            s = excess // 2
            win = win[s:s + desired_len, :]
        beats.append(win)
    if len(beats) == 0:
        return np.zeros((0, desired_len, 12), dtype=np.float32)
    return np.stack(beats).astype(np.float32)  # (n_beats, beat_len, 12)


class BeatDataset(Dataset):
    """
    Lazy Beat Dataset: given a directory of .npy files, yields beat windows.
    Each .npy is expected to be (12, T) or (T, 12). Beat extraction is done on the fly.

    NOTE: This is simple and convenient for prototyping. For large datasets, you may
    precompute beat windows and store them on disk.
    """
    def __init__(self, root_dir: str | Path, fs: int = 500, pre_r: float = 0.2, post_r: float = 0.4, beat_len: int = 400, max_beats_per_file: int = None):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"BeatDataset root not found: {self.root_dir}")
        self.files = sorted([p for p in self.root_dir.iterdir() if p.suffix == '.npy'])
        self.fs = fs
        self.pre_r = pre_r
        self.post_r = post_r
        self.beat_len = beat_len
        self.max_beats_per_file = max_beats_per_file

        # Build an index mapping (file_idx, beat_idx)
        self.index = []
        # to avoid heavy upfront work, we pre-scan file counts but do not load signals
        for fi, f in enumerate(self.files):
            try:
                arr = np.load(str(f))
                b = extract_beats(arr, fs=self.fs, pre_r=self.pre_r, post_r=self.post_r, beat_len=self.beat_len)
                n_beats = b.shape[0]
                if n_beats == 0:
                    continue
                if self.max_beats_per_file is None:
                    beats_to_add = range(n_beats)
                else:
                    beats_to_add = range(min(n_beats, self.max_beats_per_file))
                for bi in beats_to_add:
                    self.index.append((fi, bi))
            except Exception as e:
                # skip unreadable files
                print(f"[BeatDataset] skipping {f.name}: {e}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, beat_idx = self.index[idx]
        fpath = self.files[file_idx]
        arr = np.load(str(fpath))
        beats = extract_beats(arr, fs=self.fs, pre_r=self.pre_r, post_r=self.post_r, beat_len=self.beat_len)
        if beats.shape[0] == 0:
            # fallback: return zeros
            return torch.zeros((self.beat_len, self.n_leads), dtype=torch.float32)
        b = beats[beat_idx % beats.shape[0]]  # (T, 12)
        # ensure dtype and shape
        return torch.from_numpy(b).float()  # (T, 12)


# --------------------------
# Example usage & integration instructions:
# --------------------------
# 1) Save this file as src/models/beat_vae.py
# 2) Instantiate:
#    from src.models.beat_vae import BeatVAE, BeatDataset
#    model = BeatVAE(n_leads=12, beat_len=400, enc_hidden=128, latent_dim=16).to(device)
#
# 3) Dataloader:
#    ds = BeatDataset(root_dir='data/processed/ptbxl', fs=500, pre_r=0.2, post_r=0.4, beat_len=400)
#    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
#
# 4) Train step (adapt train_epoch in training.py):
#    for batch in loader:
#        # batch shape: (B, T, 12)
#        x = batch.to(device)
#        x_mean, x_logvar, mu, logvar = model(x)
#        loss, recon, kl = model.loss_function(x, x_mean, x_logvar, mu, logvar)
#        loss.backward(); optimizer.step()
#
# Notes:
# - Beat extraction uses Lead II. If your dataset's lead ordering differs, adjust index.
# - For speed, consider precomputing beats and saving them as .npy files and using a simpler Dataset.
# - You may prefer per-beat standardization (z-score) before training — your pipeline already has normalization.
#
