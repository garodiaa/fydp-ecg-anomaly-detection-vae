"""
Test ST-VAE anomaly detection on 50 normal and 50 abnormal samples
Verifies the function works correctly and can discriminate between normal/abnormal ECGs
"""

import torch
import numpy as np
import os
from pathlib import Path

# Import the model loading function
import sys
sys.path.append(str(Path(__file__).parent / 'src'))
from generate_reconstructions import load_model

# Import our ST-VAE anomaly computation function
from compute_st_vae_anomaly import compute_st_vae_anomaly_simple, compute_st_vae_metrics_breakdown

def test_st_vae_batch():
    """Test ST-VAE on batch of normal and abnormal samples"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 80)
    
    # Load ST-VAE model
    print("Loading ST-VAE model...")
    model = load_model('st_vae')
    model.eval()
    print(f"Model loaded successfully. Beta: {model.beta}")
    print("=" * 80)
    
    # Data paths
    normal_dir = Path("data/processed/ptbxl")
    abnormal_dir = Path("data/processed/ptbxl_abnormal")
    
    # Get sample files
    normal_files = sorted(list(normal_dir.glob("*.npy")))[:50]
    abnormal_files = sorted(list(abnormal_dir.glob("*.npy")))[:50]
    
    print(f"Testing on {len(normal_files)} normal and {len(abnormal_files)} abnormal samples")
    print("=" * 80)
    
    # Test normal samples
    normal_scores = []
    print("\nProcessing NORMAL samples...")
    for i, file_path in enumerate(normal_files):
        sig_np = np.load(file_path)
        score = compute_st_vae_anomaly_simple(model, sig_np)
        normal_scores.append(score)
        if i < 5:  # Print first 5
            print(f"  Normal {i+1:2d}: {file_path.name:30s} -> Score: {score:.2f}")
    
    print(f"  ... ({len(normal_files) - 5} more samples)")
    
    # Test abnormal samples
    abnormal_scores = []
    print("\nProcessing ABNORMAL samples...")
    for i, file_path in enumerate(abnormal_files):
        sig_np = np.load(file_path)
        score = compute_st_vae_anomaly_simple(model, sig_np)
        abnormal_scores.append(score)
        if i < 5:  # Print first 5
            print(f"  Abnormal {i+1:2d}: {file_path.name:30s} -> Score: {score:.2f}")
    
    print(f"  ... ({len(abnormal_files) - 5} more samples)")
    print("=" * 80)
    
    # Statistical analysis
    normal_scores = np.array(normal_scores)
    abnormal_scores = np.array(abnormal_scores)
    
    print("\nSTATISTICAL SUMMARY:")
    print(f"\nNormal Samples (n={len(normal_scores)}):")
    print(f"  Mean:   {normal_scores.mean():.2f}")
    print(f"  Std:    {normal_scores.std():.2f}")
    print(f"  Median: {np.median(normal_scores):.2f}")
    print(f"  Min:    {normal_scores.min():.2f}")
    print(f"  Max:    {normal_scores.max():.2f}")
    
    print(f"\nAbnormal Samples (n={len(abnormal_scores)}):")
    print(f"  Mean:   {abnormal_scores.mean():.2f}")
    print(f"  Std:    {abnormal_scores.std():.2f}")
    print(f"  Median: {np.median(abnormal_scores):.2f}")
    print(f"  Min:    {abnormal_scores.min():.2f}")
    print(f"  Max:    {abnormal_scores.max():.2f}")
    
    # Discrimination analysis
    mean_ratio = abnormal_scores.mean() / normal_scores.mean()
    median_ratio = np.median(abnormal_scores) / np.median(normal_scores)
    
    print(f"\nDISCRIMINATION ANALYSIS:")
    print(f"  Mean Ratio (Abnormal/Normal):   {mean_ratio:.3f}x")
    print(f"  Median Ratio (Abnormal/Normal): {median_ratio:.3f}x")
    
    # Check separation
    normal_q75 = np.percentile(normal_scores, 75)
    abnormal_q25 = np.percentile(abnormal_scores, 25)
    overlap = normal_q75 >= abnormal_q25
    
    print(f"\n  Normal 75th percentile:   {normal_q75:.2f}")
    print(f"  Abnormal 25th percentile: {abnormal_q25:.2f}")
    print(f"  Overlap: {'YES (distributions overlap)' if overlap else 'NO (good separation)'}")
    
    # Simple threshold analysis (using mean of means)
    threshold = (normal_scores.mean() + abnormal_scores.mean()) / 2
    normal_correct = np.sum(normal_scores < threshold)
    abnormal_correct = np.sum(abnormal_scores >= threshold)
    accuracy = (normal_correct + abnormal_correct) / (len(normal_scores) + len(abnormal_scores))
    
    print(f"\nSIMPLE THRESHOLD TEST (threshold={threshold:.2f}):")
    print(f"  Normal correctly classified:   {normal_correct}/{len(normal_scores)} ({100*normal_correct/len(normal_scores):.1f}%)")
    print(f"  Abnormal correctly classified: {abnormal_correct}/{len(abnormal_scores)} ({100*abnormal_correct/len(abnormal_scores):.1f}%)")
    print(f"  Overall Accuracy: {100*accuracy:.1f}%")
    
    # Detailed breakdown on first sample from each class
    print("\n" + "=" * 80)
    print("DETAILED METRICS BREAKDOWN (First Sample from Each Class):")
    print("=" * 80)
    
    print("\nNormal Sample 0:")
    sig_normal = np.load(normal_files[0])
    metrics_normal = compute_st_vae_metrics_breakdown(model, sig_normal)
    print(f"  File: {normal_files[0].name}")
    print(f"  MSE:          {metrics_normal['mse']:.4f}")
    print(f"  Weighted MSE: {metrics_normal['weighted_mse']:.4f}")
    print(f"  KL Divergence: {metrics_normal['kl_divergence']:.2f}")
    print(f"  NLL:          {metrics_normal['nll']:.2f}")
    print(f"  Total Score:  {metrics_normal['total_score']:.2f}")
    print(f"  Beta:         {metrics_normal['beta']}")
    
    print("\nAbnormal Sample 0:")
    sig_abnormal = np.load(abnormal_files[0])
    metrics_abnormal = compute_st_vae_metrics_breakdown(model, sig_abnormal)
    print(f"  File: {abnormal_files[0].name}")
    print(f"  MSE:          {metrics_abnormal['mse']:.4f}")
    print(f"  Weighted MSE: {metrics_abnormal['weighted_mse']:.4f}")
    print(f"  KL Divergence: {metrics_abnormal['kl_divergence']:.2f}")
    print(f"  NLL:          {metrics_abnormal['nll']:.2f}")
    print(f"  Total Score:  {metrics_abnormal['total_score']:.2f}")
    print(f"  Beta:         {metrics_abnormal['beta']}")
    
    print("\nComponent Ratios (Abnormal/Normal):")
    print(f"  MSE:          {metrics_abnormal['mse'] / metrics_normal['mse']:.3f}x")
    print(f"  Weighted MSE: {metrics_abnormal['weighted_mse'] / metrics_normal['weighted_mse']:.3f}x")
    print(f"  KL Divergence: {metrics_abnormal['kl_divergence'] / metrics_normal['kl_divergence']:.3f}x")
    print(f"  NLL:          {metrics_abnormal['nll'] / metrics_normal['nll']:.3f}x")
    print(f"  Total Score:  {metrics_abnormal['total_score'] / metrics_normal['total_score']:.3f}x")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT:")
    print("=" * 80)
    if mean_ratio > 1.2 and accuracy > 0.7:
        print("SUCCESS: ST-VAE anomaly detection is working correctly!")
        print(f"  - Clear discrimination (ratio={mean_ratio:.2f}x > 1.2)")
        print(f"  - Good accuracy ({100*accuracy:.1f}% > 70%)")
    elif mean_ratio > 1.1:
        print("MARGINAL: ST-VAE shows some discrimination but could be better")
        print(f"  - Moderate discrimination (ratio={mean_ratio:.2f}x)")
        print(f"  - Accuracy: {100*accuracy:.1f}%")
    else:
        print("FAILED: ST-VAE is not discriminating well on this batch")
        print(f"  - Poor discrimination (ratio={mean_ratio:.2f}x < 1.1)")
        print(f"  - Accuracy: {100*accuracy:.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    test_st_vae_batch()
