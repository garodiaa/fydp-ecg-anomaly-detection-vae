"""
Evaluate trained models on anomaly detection task.
Computes reconstruction errors on normal and abnormal ECGs,
determines threshold, calculates metrics, and updates comprehensive JSON.
"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
import json
import argparse
import time
from tqdm import tqdm

from data_processing.dataset import ECGDataset
from data_processing.window_dataset import WindowDataset
from models.vae_bilstm_attention import VAE as VAE_BiLSTM_Attention
from models.vae_gru import VAE as VAE_GRU
from models.vae_bilstm_mha import VAE_BILSTM_MHA
from models.hl_vae import HLVAE
from models.cae import CAE
from models.ba_vae import BeatVAE

def load_model(model_name, weights_path, latent_dim, seq_len, device):
    """Load trained model from weights"""
    print(f"Loading {model_name} model...")
    
    if model_name == 'vae_bilstm_attn':
        model = VAE_BiLSTM_Attention(n_leads=12, n_latent=latent_dim, beta=0.3)
    elif model_name == 'vae_gru':
        model = VAE_GRU(n_leads=12, n_latent=latent_dim, beta=0.3)
    elif model_name == 'vae_bilstm_mha':
        model = VAE_BILSTM_MHA(seq_len=seq_len, n_leads=12, latent_dim=latent_dim, beta=0.3)
    elif model_name == 'hlvae':
        model = HLVAE(n_leads=12, seq_len=seq_len, global_latent_dim=latent_dim, local_latent_dim=latent_dim//2, beta_g=0.3, beta_l=0.3)
    elif model_name == 'cae':
        model = CAE(in_channels=12)
    elif model_name == 'ba_vae':
        model = BeatVAE(n_leads=12, latent_dim=latent_dim, beat_len=seq_len, 
                       enc_hidden=256, n_layers=2, dropout=0.1, beta=0.3)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def compute_reconstruction_error(model, dataloader, device, model_type):
    """Compute reconstruction error for each sample"""
    model.eval()
    errors = []
    inference_times = []
    
    with torch.no_grad():
        for x in tqdm(dataloader, desc="Computing errors"):
            x = x.to(device)
            
            # Measure inference time
            start_time = time.time()
            
            if model_type == 'cae':
                x_recon = model(x.transpose(1, 2))
                x = x.transpose(1, 2)
            else:
                x_mean, x_logvar, mu, logvar, _, _ = model(x)
                x_recon = x_mean
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time / x.size(0))  # Per sample
            
            # Compute MSE per sample
            mse = torch.mean((x - x_recon) ** 2, dim=[1, 2])
            errors.extend(mse.cpu().numpy())
    
    return np.array(errors), np.mean(inference_times)

def determine_threshold(normal_errors, percentile=95):
    """Determine anomaly threshold from normal ECG errors"""
    threshold = np.percentile(normal_errors, percentile)
    return threshold

def compute_metrics(y_true, y_pred, y_scores):
    """Compute all evaluation metrics"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'confusion_matrix': {
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn)
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on anomaly detection')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['vae_bilstm_attn', 'vae_gru', 'vae_bilstm_mha', 'hlvae', 'cae', 'ba_vae'],
                       help='Model to evaluate')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to model weights (default: src/weights/<model>_weights/best_model.pt)')
    parser.add_argument('--json-path', type=str, default=None,
                       help='Path to comprehensive JSON (default: outputs/<model>/<model>_comprehensive.json)')
    parser.add_argument('--normal-dir', type=str, default='data/processed/ptbxl',
                       help='Directory with normal ECGs')
    parser.add_argument('--abnormal-dir', type=str, default='data/processed/ptbxl_abnormal',
                       help='Directory with abnormal ECGs')
    parser.add_argument('--use-windowing', action='store_true',
                       help='Use windowing (must match training configuration)')
    parser.add_argument('--window-size', type=int, default=1000, help='Window size')
    parser.add_argument('--stride', type=int, default=500, help='Stride for windows')
    parser.add_argument('--seq-len', type=int, default=1000, help='Sequence length (if not windowing)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--threshold-percentile', type=float, default=95,
                       help='Percentile of normal errors to use as threshold')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.weights is None:
        args.weights = ROOT / f"src/weights/{args.model}_weights/best_model.pt"
    else:
        args.weights = Path(args.weights)
    
    if args.json_path is None:
        args.json_path = ROOT / f"outputs/{args.model}/{args.model}_comprehensive.json"
    else:
        args.json_path = Path(args.json_path)
    
    normal_dir = ROOT / args.normal_dir
    abnormal_dir = ROOT / args.abnormal_dir
    
    # Check paths exist
    if not args.weights.exists():
        print(f"Error: Weights not found at {args.weights}")
        return
    if not args.json_path.exists():
        print(f"Error: JSON not found at {args.json_path}")
        return
    if not normal_dir.exists():
        print(f"Error: Normal dataset not found at {normal_dir}")
        return
    if not abnormal_dir.exists():
        print(f"Error: Abnormal dataset not found at {abnormal_dir}")
        return
    
    # Load comprehensive JSON
    print(f"Loading comprehensive JSON from {args.json_path}")
    with open(args.json_path, 'r') as f:
        model_data = json.load(f)
    
    # Extract configuration from JSON
    latent_dim = model_data['hyperparameters'].get('latent_dim', 32)
    seq_len = model_data['dataset'].get('effective_sequence_length', args.seq_len)
    
    print(f"\\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Windowing: {args.use_windowing}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Load model
    model = load_model(args.model, args.weights, latent_dim, seq_len, device)
    
    # Load datasets
    print(f"\\nLoading datasets...")
    if args.use_windowing:
        normal_ds = WindowDataset(
            ptbxl_dir=str(normal_dir),
            dataset='ptbxl',
            window_size=args.window_size,
            stride=args.stride,
            sample_length=5000
        )
        abnormal_ds = WindowDataset(
            ptbxl_dir=str(abnormal_dir),
            dataset='ptbxl',
            window_size=args.window_size,
            stride=args.stride,
            sample_length=5000
        )
    else:
        normal_ds = ECGDataset(
            ptbxl_dir=str(normal_dir),
            dataset='ptbxl',
            sample_length=args.seq_len
        )
        abnormal_ds = ECGDataset(
            ptbxl_dir=str(abnormal_dir),
            dataset='ptbxl',
            sample_length=args.seq_len
        )
    
    print(f"  Normal ECGs: {len(normal_ds)}")
    print(f"  Abnormal ECGs: {len(abnormal_ds)}")
    
    normal_loader = DataLoader(normal_ds, batch_size=args.batch_size, shuffle=False, 
                               num_workers=args.num_workers, pin_memory=True)
    abnormal_loader = DataLoader(abnormal_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
    
    # Compute reconstruction errors
    print(f"\\nComputing reconstruction errors on normal ECGs...")
    normal_errors, normal_inference_time = compute_reconstruction_error(
        model, normal_loader, device, args.model
    )
    
    print(f"\\nComputing reconstruction errors on abnormal ECGs...")
    abnormal_errors, abnormal_inference_time = compute_reconstruction_error(
        model, abnormal_loader, device, args.model
    )
    
    avg_inference_time = (normal_inference_time + abnormal_inference_time) / 2
    
    # Determine threshold
    threshold = determine_threshold(normal_errors, percentile=args.threshold_percentile)
    print(f"\\nAnomaly threshold ({args.threshold_percentile}th percentile): {threshold:.6f}")
    
    # Create labels and predictions
    y_true = np.concatenate([
        np.zeros(len(normal_errors)),  # Normal = 0
        np.ones(len(abnormal_errors))   # Abnormal = 1
    ])
    
    y_scores = np.concatenate([normal_errors, abnormal_errors])
    y_pred = (y_scores > threshold).astype(int)
    
    # Compute metrics
    print(f"\\nComputing evaluation metrics...")
    metrics = compute_metrics(y_true, y_pred, y_scores)
    
    print(f"\\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"\\nConfusion Matrix:")
    print(f"  TP: {metrics['confusion_matrix']['TP']}")
    print(f"  FP: {metrics['confusion_matrix']['FP']}")
    print(f"  TN: {metrics['confusion_matrix']['TN']}")
    print(f"  FN: {metrics['confusion_matrix']['FN']}")
    print(f"\\nInference Time: {avg_inference_time:.2f} ms/sample")
    print(f"{'='*60}")
    
    # Update comprehensive JSON
    model_data['evaluation'] = {
        'threshold': float(threshold),
        'threshold_percentile': args.threshold_percentile,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'auc_roc': metrics['auc_roc'],
        'auc_pr': metrics['auc_pr'],
        'support': {
            'normal': int(len(normal_errors)),
            'abnormal': int(len(abnormal_errors))
        }
    }
    
    model_data['confusion_matrix'] = metrics['confusion_matrix']
    
    model_data['latency'] = {
        'inference_time_ms_per_sample': float(avg_inference_time),
        'window_count': model_data['latency']['window_count']
    }
    
    # Save updated JSON
    print(f"\\nUpdating comprehensive JSON...")
    with open(args.json_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    print(f"Updated JSON saved to: {args.json_path}")
    
    # Save detailed results
    results_dir = args.json_path.parent
    results_path = results_dir / f"{args.model}_evaluation_results.txt"
    with open(results_path, 'w') as f:
        f.write(f"Anomaly Detection Evaluation Results\\n")
        f.write(f"{'='*60}\\n\\n")
        f.write(f"Model: {model_data['model_name']}\\n")
        f.write(f"Threshold: {threshold:.6f} ({args.threshold_percentile}th percentile)\\n\\n")
        f.write(f"Metrics:\\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\\n")
        f.write(f"  F1-Score: {metrics['f1_score']:.4f}\\n")
        f.write(f"  AUC-ROC: {metrics['auc_roc']:.4f}\\n")
        f.write(f"  AUC-PR: {metrics['auc_pr']:.4f}\\n\\n")
        f.write(f"Confusion Matrix:\\n")
        f.write(f"  True Positives (TP): {metrics['confusion_matrix']['TP']}\\n")
        f.write(f"  False Positives (FP): {metrics['confusion_matrix']['FP']}\\n")
        f.write(f"  True Negatives (TN): {metrics['confusion_matrix']['TN']}\\n")
        f.write(f"  False Negatives (FN): {metrics['confusion_matrix']['FN']}\\n\\n")
        f.write(f"Dataset:\\n")
        f.write(f"  Normal samples: {len(normal_errors)}\\n")
        f.write(f"  Abnormal samples: {len(abnormal_errors)}\\n\\n")
        f.write(f"Performance:\\n")
        f.write(f"  Inference time: {avg_inference_time:.2f} ms/sample\\n")
    print(f"Detailed results saved to: {results_path}")
    
    print(f"\\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
