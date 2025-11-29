import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from data_processing.dataset import ECGDataset
from models.ba_vae import BeatVAE
from models.vae_bilstm_attention import VAE as VAE_BiLSTM_Attn
from models.vae_bilstm_mha import VAE_BILSTM_MHA
from models.hl_vae import HLVAE
from models.vae_gru import VAE as VAE_GRU
from models.st_vae import ST_VAE

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_name):
    """Load trained model"""
    if model_name == 'ba_vae':
        model = BeatVAE(n_leads=12, beat_len=1000, enc_hidden=256, n_layers=2, 
                       latent_dim=32, beta=0.3, dropout=0.1).to(DEVICE)
        weights_path = ROOT / 'src' / 'weights' / 'ba_vae_weights' / 'best_model.pt'
    
    elif model_name == 'vae_bilstm_attn':
        model = VAE_BiLSTM_Attn(n_leads=12, n_latent=64).to(DEVICE)  # Correct: 64 not 32
        weights_path = ROOT / 'src' / 'weights' / 'best_vae_attn_model.pt'
    
    elif model_name == 'vae_bilstm_mha':
        model = VAE_BILSTM_MHA(seq_len=1000, n_leads=12, latent_dim=32).to(DEVICE)
        weights_path = ROOT / 'src' / 'weights' / 'vae_bilstm_mha_weights' / 'best_model.pt'
    
    elif model_name == 'hlvae':
        model = HLVAE(n_leads=12, seq_len=1000, global_latent_dim=32, local_latent_dim=16).to(DEVICE)  # Correct: local=16
        weights_path = ROOT / 'src' / 'weights' / 'hlvae_weights' / 'best_model.pt'
    
    elif model_name == 'vae_gru':
        model = VAE_GRU(n_leads=12, n_hidden=256, n_latent=32, n_layers=2, beta=0.3).to(DEVICE)
        weights_path = ROOT / 'src' / 'weights' / 'vae_gru_weights' / 'best_model.pt'
    
    elif model_name == 'st_vae':
        model = ST_VAE(n_leads=12, seq_len=1000, latent_dim=32, beta=0.5, 
                      freq_weight=0.3, dropout=0.2).to(DEVICE)
        weights_path = ROOT / 'src' / 'weights' / 'st_vae_weights' / 'best_model.pt'
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"WARNING: No weights found at {weights_path}")
    
    model.eval()
    return model

def get_sample_ecg(idx=100):
    """Get a sample ECG from PTB-XL database"""
    import wfdb
    import pandas as pd
    
    # Load PTB-XL database
    ptbxl_dir = ROOT / 'ptbxl'
    df = pd.read_csv(ptbxl_dir / 'ptbxl_database.csv')
    
    # Get record path
    record_path = df.iloc[idx]['filename_lr']
    record_name = str(ptbxl_dir / record_path.replace('.hea', ''))
    
    # Read ECG signal
    record = wfdb.rdrecord(record_name)
    ecg_signal = record.p_signal  # Shape: (time, leads)
    
    # Transpose to (time, leads) if needed
    if ecg_signal.shape[1] != 12:
        ecg_signal = ecg_signal.T
    
    # Take first 1000 samples
    ecg_signal = ecg_signal[:1000, :]
    
    # Z-score normalization per lead
    mean = ecg_signal.mean(axis=0, keepdims=True)
    std = ecg_signal.std(axis=0, keepdims=True) + 1e-8
    ecg_signal = (ecg_signal - mean) / std
    
    # Convert to tensor
    ecg_tensor = torch.FloatTensor(ecg_signal)
    
    # Get label
    import ast
    scp_codes = ast.literal_eval(df.iloc[idx]['scp_codes'])
    label = 'Normal' if 'NORM' in scp_codes else 'Abnormal'
    
    return ecg_tensor, label

def reconstruct_ecg(model, ecg, model_name):
    """Reconstruct ECG using the model"""
    ecg = ecg.unsqueeze(0).to(DEVICE)  # Add batch dimension
    
    with torch.no_grad():
        # All models return 6 values: x_mean, x_logvar, mu, logvar, attn_weights, lead_w
        x_mean, x_logvar, mu, logvar, attn_weights, lead_w = model(ecg)
    
    return x_mean.cpu().squeeze(0)  # Remove batch dimension

def plot_reconstruction(original, reconstructed, model_name, label, save_path):
    """Plot original vs reconstructed ECG for all 12 leads"""
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig, axes = plt.subplots(12, 1, figsize=(16, 20))
    fig.suptitle(f'{model_name.upper()} - ECG Reconstruction (Label: {label})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    original_np = original.cpu().numpy()
    reconstructed_np = reconstructed.numpy()
    
    # Calculate MSE per lead
    mse_per_lead = np.mean((original_np - reconstructed_np)**2, axis=0)
    total_mse = np.mean(mse_per_lead)
    
    for i in range(12):
        ax = axes[i]
        time = np.arange(original_np.shape[0]) / 100  # Convert to seconds (assuming 100 Hz)
        
        # Plot original and reconstructed
        ax.plot(time, original_np[:, i], label='Original', color='blue', linewidth=1.5, alpha=0.7)
        ax.plot(time, reconstructed_np[:, i], label='Reconstructed', color='red', 
                linewidth=1.5, alpha=0.7, linestyle='--')
        
        # Add lead name and MSE
        ax.set_ylabel(f'{lead_names[i]}\nMSE: {mse_per_lead[i]:.4f}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time[-1])
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)
        if i == 11:
            ax.set_xlabel('Time (seconds)', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    # Add overall MSE text
    fig.text(0.5, 0.002, f'Overall MSE: {total_mse:.6f}', 
             ha='center', fontsize=12, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved reconstruction plot to {save_path}")
    plt.close()
    
    return total_mse

def main():
    # Get a sample ECG
    print("Getting sample ECG...")
    ecg_sample, label = get_sample_ecg(idx=100)
    print(f"Sample shape: {ecg_sample.shape}, Label: {label}")
    
    models = ['st_vae']
    
    # Create output directory
    output_dir = ROOT / 'outputs' / 'reconstructions'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Processing {model_name.upper()}")
        print('='*60)
        
        try:
            # Load model
            model = load_model(model_name)
            
            # Reconstruct ECG
            print(f"Reconstructing ECG...")
            reconstructed = reconstruct_ecg(model, ecg_sample, model_name)
            
            # Plot and save
            save_path = output_dir / f'{model_name}_reconstruction.png'
            mse = plot_reconstruction(ecg_sample, reconstructed, model_name, label, save_path)
            
            results[model_name] = {
                'mse': float(mse),
                'output_path': str(save_path)
            }
            
            print(f"✓ {model_name.upper()} completed - MSE: {mse:.6f}")
            
        except Exception as e:
            print(f"✗ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
    
    # Save results summary
    results_file = output_dir / 'reconstruction_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for model_name, result in results.items():
        if 'mse' in result:
            print(f"{model_name.upper():20s} - MSE: {result['mse']:.6f}")
        else:
            print(f"{model_name.upper():20s} - ERROR: {result.get('error', 'Unknown')}")
    
    print(f"\nAll reconstruction images saved to: {output_dir}")
    print(f"Results summary saved to: {results_file}")

if __name__ == "__main__":
    main()
