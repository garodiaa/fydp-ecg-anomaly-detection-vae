"""
Full batch test on all 100 normal and 100 abnormal samples with corrected normalization
"""

import torch
import numpy as np
from pathlib import Path
from model_loader import load_model_by_name
from compute_st_vae_anomaly import compute_st_vae_reconstruction_and_anomaly, normalize_score

def test_all_samples():
    """Test on all SampleData samples"""
    
    # Load model
    print("Loading ST-VAE model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model_by_name('ST-VAE (Spatiotemporal VAE)', map_location=device)
    model.eval()
    print(f"Model loaded. Beta: {model.beta}, Device: {device}\n")
    
    sample_data_dir = Path("SampleData")
    normal_dir = sample_data_dir / "normal"
    abnormal_dir = sample_data_dir / "abnormal"
    
    threshold = 0.69
    
    # Test normal samples
    print("Testing NORMAL samples...")
    normal_scores = []
    normal_files = sorted(list(normal_dir.glob("*.npy")))
    for i, fpath in enumerate(normal_files):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(normal_files)}")
        ecg_data = np.load(fpath)
        if ecg_data.shape[0] != 12:
            ecg_data = ecg_data.T
        if ecg_data.shape[1] < 5000:
            ecg_data = np.pad(ecg_data, ((0,0),(0, 5000 - ecg_data.shape[1])), mode='constant')
        else:
            ecg_data = ecg_data[:, :5000]
        
        _, _, _, raw_score = compute_st_vae_reconstruction_and_anomaly(model, ecg_data, window_size=1000, stride=500)
        score = normalize_score(raw_score)
        normal_scores.append(score)
    
    # Test abnormal samples
    print("Testing ABNORMAL samples...")
    abnormal_scores = []
    abnormal_files = sorted(list(abnormal_dir.glob("*.npy")))
    for i, fpath in enumerate(abnormal_files):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(abnormal_files)}")
        ecg_data = np.load(fpath)
        if ecg_data.shape[0] != 12:
            ecg_data = ecg_data.T
        if ecg_data.shape[1] < 5000:
            ecg_data = np.pad(ecg_data, ((0,0),(0, 5000 - ecg_data.shape[1])), mode='constant')
        else:
            ecg_data = ecg_data[:, :5000]
        
        _, _, _, raw_score = compute_st_vae_reconstruction_and_anomaly(model, ecg_data, window_size=1000, stride=500)
        score = normalize_score(raw_score)
        abnormal_scores.append(score)
    
    # Calculate statistics
    normal_scores = np.array(normal_scores)
    abnormal_scores = np.array(abnormal_scores)
    
    normal_correct = np.sum(normal_scores < threshold)
    abnormal_correct = np.sum(abnormal_scores >= threshold)
    total_correct = normal_correct + abnormal_correct
    total_samples = len(normal_scores) + len(abnormal_scores)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"\nNormal Samples (n={len(normal_scores)}):")
    print(f"  Range: [{normal_scores.min():.4f}, {normal_scores.max():.4f}]")
    print(f"  Mean ± Std: {normal_scores.mean():.4f} ± {normal_scores.std():.4f}")
    print(f"  Correctly classified: {normal_correct}/{len(normal_scores)} ({100*normal_correct/len(normal_scores):.1f}%)")
    
    print(f"\nAbnormal Samples (n={len(abnormal_scores)}):")
    print(f"  Range: [{abnormal_scores.min():.4f}, {abnormal_scores.max():.4f}]")
    print(f"  Mean ± Std: {abnormal_scores.mean():.4f} ± {abnormal_scores.std():.4f}")
    print(f"  Correctly classified: {abnormal_correct}/{len(abnormal_scores)} ({100*abnormal_correct/len(abnormal_scores):.1f}%)")
    
    print(f"\nOverall Accuracy: {total_correct}/{total_samples} ({100*total_correct/total_samples:.2f}%)")
    print(f"Threshold: {threshold}")
    print("="*70)

if __name__ == "__main__":
    with torch.no_grad():
        test_all_samples()
