"""
Hierarchical Latent VAE for 12-lead ECG anomaly detection.

Design:
 - Global encoder compresses sequence-level info -> z_g (shape: [B, Dg])
 - Local encoder produces per-timestep latents -> z_l (shape: [B, T, Dl])
 - Decoder conditions reconstruction on concat([z_g_expanded, z_l]) per time step
 - Loss: negative log-likelihood under Normal(x_mean, exp(0.5*x_logvar)) + beta_g * KL_g + beta_l * KL_l

Inputs: x (B, T, n_leads)  — matches existing datasets.
Outputs: x_mean, x_logvar, mu_g, logvar_g, mu_l, logvar_l
"""
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class HLVAE(nn.Module):
    def __init__(
        self,
        n_leads: int = 12,
        seq_len: int = 500,
        global_latent_dim: int = 64,
        local_latent_dim: int = 16,
        enc_hidden_global: tuple = (512, 256),
        enc_hidden_local: int = 128,
        dec_hidden: tuple = (256, 512),
        beta_g: float = 0.01,
        beta_l: float = 0.01,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_leads = n_leads
        self.seq_len = seq_len
        self.global_latent_dim = global_latent_dim
        self.local_latent_dim = local_latent_dim
        self.beta_g = beta_g
        self.beta_l = beta_l

        # Global encoder (2-layer BiLSTM)
        self.enc_gl_1 = nn.LSTM(
            input_size=n_leads,
            hidden_size=enc_hidden_global[0],
            batch_first=True,
            bidirectional=True,
        )
        self.enc_gl_2 = nn.LSTM(
            input_size=2 * enc_hidden_global[0],
            hidden_size=enc_hidden_global[1],
            batch_first=True,
            bidirectional=True,
        )
        enc_gl_out = 2 * enc_hidden_global[1]
        self.to_stats_g = nn.Sequential(
            nn.Linear(enc_gl_out, enc_gl_out // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_gl_out // 2, 2 * global_latent_dim),
        )

        # Local encoder (BiLSTM per timestep -> stats)
        self.enc_loc = nn.LSTM(
            input_size=n_leads,
            hidden_size=enc_hidden_local,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        enc_loc_out = 2 * enc_hidden_local
        self.to_stats_l = nn.Sequential(
            nn.Linear(enc_loc_out, enc_loc_out // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_loc_out // 2, 2 * local_latent_dim),
        )

        # Decoder
        dec_in = local_latent_dim + global_latent_dim
        self.dec_lstm1 = nn.LSTM(
            input_size=dec_in,
            hidden_size=dec_hidden[0],
            batch_first=True,
            bidirectional=True,
        )
        self.dec_lstm2 = nn.LSTM(
            input_size=2 * dec_hidden[0],
            hidden_size=dec_hidden[1],
            batch_first=True,
            bidirectional=True,
        )
        dec_out_dim = 2 * dec_hidden[1]
        self.to_recon = nn.Sequential(
            nn.Linear(dec_out_dim, dec_out_dim // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_out_dim // 2, 2 * n_leads),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def encode_global(self, x):
        h, _ = self.enc_gl_1(x)
        h, _ = self.enc_gl_2(h)
        h_pool = h.mean(dim=1)
        stats = self.to_stats_g(h_pool)
        mu_g, logvar_g = stats.chunk(2, dim=-1)
        return mu_g, logvar_g

    def encode_local(self, x):
        h, _ = self.enc_loc(x)
        B, T, C = h.shape
        h_flat = h.reshape(B * T, C)
        stats = self.to_stats_l(h_flat)
        mu_l, logvar_l = stats.chunk(2, dim=-1)
        mu_l = mu_l.reshape(B, T, -1)
        logvar_l = logvar_l.reshape(B, T, -1)
        return mu_l, logvar_l

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_g, z_l):
        B, T, Dl = z_l.shape
        z_g_exp = z_g.unsqueeze(1).expand(-1, T, -1)
        z_comb = torch.cat([z_g_exp, z_l], dim=-1)
        d, _ = self.dec_lstm1(z_comb)
        d, _ = self.dec_lstm2(d)
        rec_stats = self.to_recon(d.reshape(B * T, -1))
        x_mean, x_logvar = rec_stats.chunk(2, dim=-1)
        x_mean = x_mean.reshape(B, T, self.n_leads)
        x_logvar = x_logvar.reshape(B, T, self.n_leads)
        return x_mean, x_logvar

    def forward(self, x):
        mu_g, logvar_g = self.encode_global(x)
        z_g = self.reparameterize(mu_g, logvar_g)
        mu_l, logvar_l = self.encode_local(x)
        z_l = self.reparameterize(mu_l, logvar_l)
        x_mean, x_logvar = self.decode(z_g, z_l)
        # Store for loss computation
        self._mu_g = mu_g
        self._logvar_g = logvar_g
        self._mu_l = mu_l
        self._logvar_l = logvar_l
        # Create attention placeholders for compatibility
        B, T, C = x.shape
        attn_weights = torch.ones(B, T, T, device=x.device) / T
        lead_w = torch.ones(B, T, C, device=x.device) / C
        return x_mean, x_logvar, mu_l, logvar_l, attn_weights, lead_w

    def loss_function(self, x, x_mean, x_logvar, mu_l, logvar_l):
        """Standard interface - uses stored global latents"""
        x_logvar = torch.clamp(x_logvar, min=-10.0, max=10.0)
        dist = Normal(x_mean, torch.exp(0.5 * x_logvar))
        log_px = dist.log_prob(x)
        recon_loss = -log_px.sum(dim=[1, 2]).mean()
        # Use stored global latents
        mu_g, logvar_g = self._mu_g, self._logvar_g
        kl_g = -0.5 * (1 + logvar_g - mu_g.pow(2) - logvar_g.exp())
        kl_g = kl_g.sum(dim=-1).mean()
        kl_l = -0.5 * (1 + logvar_l - mu_l.pow(2) - logvar_l.exp())
        kl_l = kl_l.sum(dim=-1).mean()
        # Combined KL for standard interface
        kl_loss = kl_g + kl_l
        total = recon_loss + self.beta_g * kl_g + self.beta_l * kl_l
        return total, recon_loss, kl_loss
