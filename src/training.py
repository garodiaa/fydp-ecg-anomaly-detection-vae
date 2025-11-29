from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import random
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, plots will not be generated")
from data_processing.dataset import ECGDataset
from data_processing.window_dataset import WindowDataset
from models.vae_bilstm_attention import VAE as VAE_BiLSTM_Attention
from models.vae_gru import VAE as VAE_GRU
from models.vae_bilstm_mha import VAE_BILSTM_MHA
from models.hl_vae import HLVAE
from models.cae import CAE
from models.ba_vae import BeatVAE
from models.transformer_vae import TransformerVAE
from models.diffusion_vae import DiffusionVAE
from models.hierarchical_attention_vae import HierarchicalAttentionVAE
from models.st_vae import ST_VAE
import argparse
import json

TQDM_CONFIG = {"dynamic_ncols": True, "leave": False, "ascii": True, "mininterval": 0.1}

def cyclical_annealing_beta(epoch, cycle_period=10, ramp_ratio=0.5, max_beta=1.0):
    cycle_epoch = (epoch - 1) % cycle_period
    ramp_epochs = max(1, int(cycle_period * ramp_ratio))
    if cycle_epoch < ramp_epochs:
        return max_beta * (cycle_epoch + 1) / ramp_epochs
    return max_beta

def train_epoch(model, dataloader, optimizer, device, model_type, desc="Train"):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    loader_tqdm = tqdm(dataloader, desc=desc, **TQDM_CONFIG)
    for x in loader_tqdm:
        x = x.to(device)
        optimizer.zero_grad()
        if model_type == 'cae':
            x_recon = model(x.transpose(1, 2))
            x = x.transpose(1, 2)
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(x_recon, x)
            recon_loss = loss
            kl_loss = 0.0
        else:
            x_mean, x_logvar, mu, logvar, _, _ = model(x)
            # Handle DiffusionVAE which needs compute_diffusion parameter
            if isinstance(model, DiffusionVAE):
                loss, recon_loss, kl_loss = model.loss_function(x, x_mean, x_logvar, mu, logvar, compute_diffusion=True)
            else:
                loss, recon_loss, kl_loss = model.loss_function(x, x_mean, x_logvar, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_size = x.size(0)
        total_loss  += loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_kl    += kl_loss if isinstance(kl_loss, float) else kl_loss.item() * batch_size
    n_samples = len(dataloader.dataset)
    return total_loss / n_samples, total_recon / n_samples, total_kl / n_samples

def validate_epoch(model, dataloader, device, model_type, desc="Validate"):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    loader_tqdm = tqdm(dataloader, desc=desc, **TQDM_CONFIG)
    with torch.no_grad():
        for x in loader_tqdm:
            x = x.to(device)
            if model_type == 'cae':
                x_recon = model(x.transpose(1, 2))
                x = x.transpose(1, 2)
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(x_recon, x)
                recon_loss = loss
                kl_loss = 0.0
            else:
                x_mean, x_logvar, mu, logvar, _, _ = model(x)
                # Handle DiffusionVAE which needs compute_diffusion parameter (but not during validation)
                if isinstance(model, DiffusionVAE):
                    loss, recon_loss, kl_loss = model.loss_function(x, x_mean, x_logvar, mu, logvar, compute_diffusion=False)
                else:
                    loss, recon_loss, kl_loss = model.loss_function(x, x_mean, x_logvar, mu, logvar)
            batch_size = x.size(0)
            total_loss  += loss.item() * batch_size
            total_recon += recon_loss.item() * batch_size
            total_kl    += kl_loss if isinstance(kl_loss, float) else kl_loss.item() * batch_size
    n_samples = len(dataloader.dataset)
    return total_loss / n_samples, total_recon / n_samples, total_kl / n_samples

def compute_vae_diagnostics(model, dataloader, device):
    model.eval()
    all_mu = []
    all_logvar = []
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            x_mean, x_logvar, mu, logvar, _, _ = model(x)
            B, T, D = mu.shape
            all_mu.append(mu.reshape(B * T, D).cpu())
            all_logvar.append(logvar.reshape(B * T, D).cpu())
    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    post_var = all_logvar.exp().mean(dim=0)
    mean_mu  = all_mu.mean(dim=0)
    kl_per = -0.5 * (1 + all_logvar - all_mu.pow(2) - all_logvar.exp())
    kl_per_dim = kl_per.mean(dim=0)
    low_kl_dims = (kl_per_dim < 1e-3).sum().item()
    total_dims = kl_per_dim.numel()
    return {'post_var': post_var, 'mean_mu': mean_mu, 'kl_per_dim': kl_per_dim, 'low_kl_dims': low_kl_dims, 'total_dims': total_dims}

def plot_training_curves(history, save_dir, model_name):
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping plot generation")
        return
    epochs = list(range(1, len(history['train_loss']) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} Training Curves', fontsize=16)
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 1].plot(epochs, history['train_recon'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_recon'], 'r-', label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    if 'train_kl' in history:
        axes[1, 0].plot(epochs, history['train_kl'], 'b-', label='Train')
        axes[1, 0].plot(epochs, history['val_kl'], 'r-', label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    if 'lr' in history:
        axes[1, 1].plot(epochs, history['lr'], 'g-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_metrics_summary(history, save_dir, model_name):
    with open(save_dir / f'{model_name}_metrics.txt', 'w') as f:
        f.write(f"Training Summary for {model_name}\\n")
        f.write("=" * 60 + "\\n\\n")
        best_epoch = np.argmin(history['val_loss']) + 1
        f.write(f"Best Validation Loss: {min(history['val_loss']):.6f} at Epoch {best_epoch}\\n")
        f.write(f"Final Train Loss: {history['train_loss'][-1]:.6f}\\n")
        f.write(f"Final Val Loss: {history['val_loss'][-1]:.6f}\\n\\n")
        if 'val_kl' in history:
            f.write(f"Final Train Recon: {history['train_recon'][-1]:.6f}\\n")
            f.write(f"Final Val Recon: {history['val_recon'][-1]:.6f}\\n")
            f.write(f"Final Train KL: {history['train_kl'][-1]:.6f}\\n")
            f.write(f"Final Val KL: {history['val_kl'][-1]:.6f}\\n\\n")
        f.write("\\nEpoch-by-Epoch Details:\\n")
        f.write("-" * 60 + "\\n")
        for i, epoch in enumerate(range(1, len(history['train_loss']) + 1)):
            f.write(f"Epoch {epoch:3d}: Train Loss={history['train_loss'][i]:.6f}, Val Loss={history['val_loss'][i]:.6f}")
            if 'val_kl' in history:
                f.write(f", Val KL={history['val_kl'][i]:.6f}")
            f.write("\\n")

def save_comprehensive_json(args, model, history, train_ds_size, val_ds_size, save_dir, model_name):
    """Save comprehensive model metadata JSON for visualization and analysis"""
    
    # Get model-specific architecture info
    model_display_names = {
        'vae_bilstm_attn': 'VAE BiLSTM with Attention',
        'vae_gru': 'VAE with GRU',
        'vae_bilstm_mha': 'VAE BiLSTM with Multi-Head Attention',
        'hlvae': 'Hierarchical Latent VAE',
        'cae': 'Convolutional Autoencoder',
        'ba_vae': 'Beat-Aligned VAE'
    }
    
    model_notes = {
        'vae_bilstm_attn': 'BiLSTM encoder-decoder with lead-wise and temporal attention mechanisms',
        'vae_gru': 'GRU-based variational autoencoder for temporal ECG modeling',
        'vae_bilstm_mha': 'BiLSTM with multi-head attention (MA-VAE style)',
        'hlvae': 'Hierarchical VAE with global and local latent representations',
        'cae': 'Convolutional autoencoder baseline for ECG reconstruction',
        'ba_vae': 'Beat-aligned VAE with bidirectional GRU encoder, uses per-beat latent representation'
    }
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Determine sequence configuration
    if args.use_windowing:
        seq_config = {
            "mode": "windowing",
            "window_size": args.window_size,
            "stride": args.stride,
            "sample_length": 5000,
            "effective_sequence_length": args.window_size
        }
    else:
        seq_config = {
            "mode": "fixed_length",
            "window_size": args.seq_len,
            "stride": None,
            "sample_length": args.seq_len,
            "effective_sequence_length": args.seq_len
        }
    
    # Build comprehensive JSON structure
    model_data = {
        "model_name": model_display_names.get(model_name, model_name),
        "model_id": model_name,
        
        "dataset": {
            "train_dataset": "PTB-XL",
            "test_dataset": "PTB-XL",
            "train_samples": train_ds_size,
            "val_samples": val_ds_size,
            "n_leads": 12,
            **seq_config
        },
        
        "training": {
            "epochs": args.epochs,
            "train_loss": [float(x) for x in history['train_loss']],
            "val_loss": [float(x) for x in history['val_loss']],
            "train_recon": [float(x) for x in history['train_recon']],
            "val_recon": [float(x) for x in history['val_recon']],
            "train_kl": [float(x) for x in history.get('train_kl', [])],
            "val_kl": [float(x) for x in history.get('val_kl', [])],
            "lr": [float(x) for x in history['lr']],
            "beta": args.beta if model_name != 'cae' else None,
            "best_epoch": int(np.argmin(history['val_loss']) + 1),
            "best_val_loss": float(min(history['val_loss']))
        },
        
        "evaluation": {
            "threshold": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "auc_roc": 0.0,
            "auc_pr": 0.0,
            "support": {
                "normal": 0,
                "abnormal": 0
            },
            "note": "Evaluation metrics to be filled after testing on anomaly detection dataset"
        },
        
        "confusion_matrix": {
            "TP": 0,
            "FP": 0,
            "TN": 0,
            "FN": 0,
            "note": "Confusion matrix to be filled after testing"
        },
        
        "latency": {
            "inference_time_ms_per_sample": 0.0,
            "window_count": (5000 - args.window_size) // args.stride + 1 if args.use_windowing else 1,
            "note": "Inference time to be measured during evaluation"
        },
        
        "hyperparameters": {
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim if model_name != 'cae' else None,
            "beta": args.beta if model_name != 'cae' else None,
            "beta_annealing": {
                "cycle_period": 10,
                "ramp_ratio": 0.5,
                "max_beta": args.beta
            } if model_name != 'cae' else None,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_params": {
                "factor": 0.5,
                "patience": 5
            },
            "gradient_clipping": {
                "max_norm": 1.0
            },
            "num_workers": args.num_workers
        },
        
        "architecture_info": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "notes": model_notes.get(model_name, "")
        },
        
        "training_config": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
        }
    }
    
    # Save comprehensive JSON
    json_path = save_dir / f'{model_name}_comprehensive.json'
    with open(json_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    return json_path

def main():
    p = argparse.ArgumentParser(description='Train ECG anomaly detection models on PTB-XL')
    p.add_argument('--model', type=str, choices=['vae_bilstm_attn', 'vae_gru', 'vae_bilstm_mha', 'hlvae', 'cae', 'ba_vae', 'transformer_vae', 'diffusion_vae', 'ha_vae', 'st_vae'], required=True, help='Model to train')
    p.add_argument('--dataset', type=str, default='ptbxl', help='Dataset name (ptbxl only)')
    p.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    p.add_argument('--batch_size', type=int, default=64, help='Batch size')
    p.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    p.add_argument('--latent_dim', type=int, default=32, help='Latent dimension')
    p.add_argument('--beta', type=float, default=0.3, help='Beta for KL annealing (max value)')
    p.add_argument('--reduce_train', type=int, default=None, help='Limit training samples (None=all)')
    p.add_argument('--reduce_val', type=int, default=None, help='Limit validation samples (None=all)')
    p.add_argument('--seq_len', type=int, default=1000, help='ECG sequence length when not using windowing')
    p.add_argument('--use_windowing', action='store_true', help='Use sliding windows on full 5000-sample ECGs')
    p.add_argument('--window_size', type=int, default=1000, help='Window size when using windowing')
    p.add_argument('--stride', type=int, default=500, help='Stride for sliding windows')
    p.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (0 for Windows)')
    args = p.parse_args()

    DATA_DIR = ROOT / "data" / "processed"
    OUTPUT_DIR = ROOT / "outputs" / args.model
    MODEL_DIR = ROOT / "src" / "weights" / f"{args.model}_weights"
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    os.makedirs(str(MODEL_DIR), exist_ok=True)

    CYCLE_PERIOD = 10
    RAMP_RATIO = 0.5
    MAX_BETA = args.beta

    if args.use_windowing:
        print(f"\nLoading PTB-XL dataset with windowing...")
        print(f"Window size: {args.window_size}, Stride: {args.stride}, Full length: 5000")
        full_ds = WindowDataset(
            ptbxl_dir=str(DATA_DIR / "ptbxl"),
            dataset='ptbxl',
            window_size=args.window_size,
            stride=args.stride,
            sample_length=5000
        )
    else:
        print(f"\nLoading PTB-XL dataset (fixed length {args.seq_len})...")
        full_ds = ECGDataset(
            ptbxl_dir=str(DATA_DIR / "ptbxl"),
            dataset='ptbxl',
            sample_length=args.seq_len
        )

    indices = list(range(len(full_ds)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    if args.reduce_train is not None:
        train_idx = random.sample(train_idx, min(args.reduce_train, len(train_idx)))
    if args.reduce_val is not None:
        val_idx = random.sample(val_idx, min(args.reduce_val, len(val_idx)))
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\\n" + "=" * 60)
    print("GPU STATUS REPORT:")
    print("=" * 60)
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Device: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Training Device: {device}")
    print("=" * 60)

    model_seq_len = args.window_size if args.use_windowing else args.seq_len
    print(f"\nInitializing {args.model} model (sequence length: {model_seq_len})...")
    if args.model == 'vae_bilstm_attn':
        model = VAE_BiLSTM_Attention(n_leads=12, n_latent=args.latent_dim, beta=args.beta).to(device)
    elif args.model == 'vae_gru':
        model = VAE_GRU(n_leads=12, n_latent=args.latent_dim, beta=args.beta).to(device)
    elif args.model == 'vae_bilstm_mha':
        model = VAE_BILSTM_MHA(seq_len=model_seq_len, n_leads=12, latent_dim=args.latent_dim, beta=args.beta).to(device)
    elif args.model == 'hlvae':
        model = HLVAE(n_leads=12, seq_len=model_seq_len, global_latent_dim=args.latent_dim, local_latent_dim=args.latent_dim//2, beta_g=args.beta, beta_l=args.beta).to(device)
    elif args.model == 'cae':
        model = CAE(in_channels=12).to(device)
    elif args.model == 'ba_vae':
        # BeatVAE: beat-aligned VAE with R-peak segmentation (improved architecture)
        model = BeatVAE(n_leads=12, latent_dim=args.latent_dim, beat_len=model_seq_len, 
                       enc_hidden=256, n_layers=2, dropout=0.1, beta=args.beta).to(device)
        print(f"   BeatVAE configured: beat_len={model_seq_len}, enc_hidden=256, layers=2, latent_dim={args.latent_dim}")
    elif args.model == 'transformer_vae':
        # Transformer-VAE: Self-attention based encoder/decoder
        model = TransformerVAE(
            n_leads=12,
            seq_len=model_seq_len,
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            latent_dim=args.latent_dim,
            beta=args.beta,
            dropout=0.1
        ).to(device)
        print(f"   Transformer-VAE configured: seq_len={model_seq_len}, d_model=256, heads=8, layers=4+4, latent_dim={args.latent_dim}")
    elif args.model == 'diffusion_vae':
        # Diffusion-VAE: VAE with diffusion process
        model = DiffusionVAE(
            n_leads=12,
            seq_len=model_seq_len,
            hidden_dim=256,
            latent_dim=args.latent_dim,
            num_layers=2,
            beta=args.beta,
            diffusion_steps=100,
            diffusion_weight=0.5,
            dropout=0.1
        ).to(device)
        print(f"   Diffusion-VAE configured: seq_len={model_seq_len}, hidden_dim=256, diffusion_steps=100, latent_dim={args.latent_dim}")
    elif args.model == 'ha_vae':
        # Hierarchical Attention VAE: Per-lead + cross-lead attention
        model = HierarchicalAttentionVAE(
            n_leads=12,
            seq_len=model_seq_len,
            hidden_dim=128,
            global_latent_dim=args.latent_dim,
            local_latent_dim=args.latent_dim // 2,
            num_heads=4,
            beta=args.beta,
            dropout=0.1
        ).to(device)
        print(f"   Hierarchical Attention VAE configured: seq_len={model_seq_len}, hidden_dim=128, heads=4, latent_dim={args.latent_dim}")
    
    elif args.model == 'st_vae':
        # Spectral-Temporal Disentangled VAE: Time + Frequency domain
        model = ST_VAE(
            n_leads=12,
            seq_len=model_seq_len,
            latent_dim=args.latent_dim,
            beta=args.beta,
            freq_weight=0.3,
            dropout=0.2
        ).to(device)
        print(f"   ST-VAE configured: seq_len={model_seq_len}, latent_dim={args.latent_dim}, freq_weight=0.3, beta={args.beta}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    history = {'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': [], 'train_kl': [], 'val_kl': [], 'lr': []}
    best_val_loss = float('inf')

    print(f"\\nStarting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        if args.model != 'cae':
            if hasattr(model, 'beta'):
                model.beta = cyclical_annealing_beta(epoch, CYCLE_PERIOD, RAMP_RATIO, MAX_BETA)
            if hasattr(model, 'beta_g'):
                beta_val = cyclical_annealing_beta(epoch, CYCLE_PERIOD, RAMP_RATIO, MAX_BETA)
                model.beta_g = beta_val
                model.beta_l = beta_val
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device, args.model, desc=f"Epoch {epoch:03d}/{args.epochs:03d} Train")
        val_loss, val_recon, val_kl = validate_epoch(model, val_loader, device, args.model, desc=f"Epoch {epoch:03d}/{args.epochs:03d} Val")
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon'].append(train_recon)
        history['val_recon'].append(val_recon)
        history['train_kl'].append(train_kl)
        history['val_kl'].append(val_kl)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        if args.model != 'cae':
            diag = compute_vae_diagnostics(model, val_loader, device)
            train_ratio = train_kl / train_recon if train_recon > 0 else float('inf')
            val_ratio = val_kl / val_recon if val_recon > 0 else float('inf')
            beta_val = model.beta if hasattr(model, 'beta') else model.beta_g
            tqdm.write(f"Epoch {epoch:03d} → Train NLL={train_recon:.4f}, Train KL={train_kl:.4f}, KL/NLL={train_ratio:.2f} | Val NLL={val_recon:.4f}, Val KL={val_kl:.4f}, KL/NLL={val_ratio:.2f} | Low-KL dims={diag['low_kl_dims']}/{diag['total_dims']} | Post Var={diag['post_var'].mean():.4f}, Mu Var={diag['mean_mu'].var():.4f} | LR={optimizer.param_groups[0]['lr']:.6f} | Beta={beta_val:.4f}")
        else:
            tqdm.write(f"Epoch {epoch:03d} → Train MSE={train_recon:.4f}, Val MSE={val_recon:.4f}")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_max_memory = torch.cuda.max_memory_allocated() / 1024**3
            tqdm.write(f"         GPU Memory: {gpu_memory:.2f}GB used, {gpu_max_memory:.2f}GB peak")
        scheduler.step(val_loss)
        epoch_model_path = MODEL_DIR / f"model_epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), epoch_model_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = MODEL_DIR / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            tqdm.write(f"         ★ Best model saved (val_loss={val_loss:.6f})")

    print("\\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    history_path = OUTPUT_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        history_serializable = {k: [float(x) for x in v] for k, v in history.items()}
        json.dump(history_serializable, f, indent=2)
    print(f"\\nTraining history saved to {history_path}")

    print("Generating training curves...")
    plot_training_curves(history, OUTPUT_DIR, args.model)
    print(f"Training curves saved to {OUTPUT_DIR / f'{args.model}_training_curves.png'}")

    save_metrics_summary(history, OUTPUT_DIR, args.model)
    print(f"Metrics summary saved to {OUTPUT_DIR / f'{args.model}_metrics.txt'}")

    print("Saving comprehensive model JSON...")
    json_path = save_comprehensive_json(
        args, model, history, len(train_ds), len(val_ds), OUTPUT_DIR, args.model
    )
    print(f"Comprehensive JSON saved to {json_path}")

    print("\\n" + "=" * 60)
    print("FINAL EVALUATION METRICS")
    print("=" * 60)
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
    if args.model != 'cae':
        print(f"Final Train Reconstruction: {history['train_recon'][-1]:.6f}")
        print(f"Final Val Reconstruction: {history['val_recon'][-1]:.6f}")
        print(f"Final Train KL: {history['train_kl'][-1]:.6f}")
        print(f"Final Val KL: {history['val_kl'][-1]:.6f}")
    print("\\n" + "=" * 60)
    print(f"Model weights saved to {MODEL_DIR}")
    print(f"Outputs saved to {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
