"""
Test ST-VAE anomaly detection on 50 normal and 50 abnormal samples
Verifies the function works correctly and can discriminate between normal/abnormal ECGs
"""

import torch
import numpy as np
from pathlib import Path

# Import the model loading function
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_reconstructions import load_model

# Import our ST-VAE anomaly computation function
from anomaly.compute_st_vae_anomaly import compute_st_vae_anomaly_simple, compute_st_vae_metrics_breakdown

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
    normal_dir = PROJECT_ROOT / "data/demo_samples/normal"
    abnormal_dir = PROJECT_ROOT / "data/demo_samples/abnormal"
    
    # Get sample files
    normal_files = sorted(list(normal_dir.glob("*.npy")))[:50]
    abnormal_files = sorted(list(abnormal_dir.glob("*.npy")))[:50]
    
    print(f"Testing on {len(normal_files)} normal and {len(abnormal_files)} abnormal samples")
    print("=" * 80)
    
    # Test normal samples
    normal_scores = []
    normal_scores_normalized = []
    print("\nProcessing NORMAL samples...")
    for i, file_path in enumerate(normal_files):
        sig_np = np.load(file_path)
        score_raw = compute_st_vae_anomaly_simple(model, sig_np, normalized=False)
        score_norm = compute_st_vae_anomaly_simple(model, sig_np, normalized=True)
        normal_scores.append(score_raw)
        normal_scores_normalized.append(score_norm)
        if i < 5:  # Print first 5
            print(f"  Normal {i+1:2d}: {file_path.name:30s} -> Raw: {score_raw:.2f}, Normalized: {score_norm:.4f}")
    
    print(f"  ... ({len(normal_files) - 5} more samples)")
    
    # Test abnormal samples
    abnormal_scores = []
    abnormal_scores_normalized = []
    print("\nProcessing ABNORMAL samples...")
    for i, file_path in enumerate(abnormal_files):
        sig_np = np.load(file_path)
        score_raw = compute_st_vae_anomaly_simple(model, sig_np, normalized=False)
        score_norm = compute_st_vae_anomaly_simple(model, sig_np, normalized=True)
        abnormal_scores.append(score_raw)
        abnormal_scores_normalized.append(score_norm)
        if i < 5:  # Print first 5
            print(f"  Abnormal {i+1:2d}: {file_path.name:30s} -> Raw: {score_raw:.2f}, Normalized: {score_norm:.4f}")
    
    print(f"  ... ({len(abnormal_files) - 5} more samples)")
    print("=" * 80)
    
    # Statistical analysis
    normal_scores = np.array(normal_scores)
    abnormal_scores = np.array(abnormal_scores)
    normal_scores_normalized = np.array(normal_scores_normalized)
    abnormal_scores_normalized = np.array(abnormal_scores_normalized)
    
    print("\nSTATISTICAL SUMMARY:")
    print("\n" + "=" * 80)
    print("RAW SCORES (Internal Calculation)")
    print("=" * 80)
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
    
    print("\n" + "=" * 80)
    print("NORMALIZED SCORES (For Presentation - Range [0, 1])")
    print("=" * 80)
    print(f"\nNormal Samples (n={len(normal_scores_normalized)}):")
    print(f"  Mean:   {normal_scores_normalized.mean():.4f}")
    print(f"  Std:    {normal_scores_normalized.std():.4f}")
    print(f"  Median: {np.median(normal_scores_normalized):.4f}")
    print(f"  Min:    {normal_scores_normalized.min():.4f}")
    print(f"  Max:    {normal_scores_normalized.max():.4f}")
    
    print(f"\nAbnormal Samples (n={len(abnormal_scores_normalized)}):")
    print(f"  Mean:   {abnormal_scores_normalized.mean():.4f}")
    print(f"  Std:    {abnormal_scores_normalized.std():.4f}")
    print(f"  Median: {np.median(abnormal_scores_normalized):.4f}")
    print(f"  Min:    {abnormal_scores_normalized.min():.4f}")
    print(f"  Max:    {abnormal_scores_normalized.max():.4f}")
    
    # Discrimination analysis
    mean_ratio = abnormal_scores.mean() / normal_scores.mean()
    median_ratio = np.median(abnormal_scores) / np.median(normal_scores)
    mean_ratio_norm = abnormal_scores_normalized.mean() / normal_scores_normalized.mean()
    median_ratio_norm = np.median(abnormal_scores_normalized) / np.median(normal_scores_normalized)
    
    print(f"\nDISCRIMINATION ANALYSIS:")
    print(f"  Raw Scores:")
    print(f"    Mean Ratio (Abnormal/Normal):   {mean_ratio:.3f}x")
    print(f"    Median Ratio (Abnormal/Normal): {median_ratio:.3f}x")
    print(f"  Normalized Scores:")
    print(f"    Mean Ratio (Abnormal/Normal):   {mean_ratio_norm:.3f}x")
    print(f"    Median Ratio (Abnormal/Normal): {median_ratio_norm:.3f}x")
    
    # Check separation (using normalized scores for presentation)
    normal_q75 = np.percentile(normal_scores_normalized, 75)
    abnormal_q25 = np.percentile(abnormal_scores_normalized, 25)
    overlap = normal_q75 >= abnormal_q25
    
    print(f"\n  Normalized Score Separation:")
    print(f"    Normal 75th percentile:   {normal_q75:.4f}")
    print(f"    Abnormal 25th percentile: {abnormal_q25:.4f}")
    print(f"    Overlap: {'YES (distributions overlap)' if overlap else 'NO (good separation)'}")
    
    # Simple threshold analysis (using normalized scores)
    threshold_norm = (normal_scores_normalized.mean() + abnormal_scores_normalized.mean()) / 2
    normal_correct = np.sum(normal_scores_normalized < threshold_norm)
    abnormal_correct = np.sum(abnormal_scores_normalized >= threshold_norm)
    accuracy = (normal_correct + abnormal_correct) / (len(normal_scores_normalized) + len(abnormal_scores_normalized))
    
    print(f"\nSIMPLE THRESHOLD TEST (normalized threshold={threshold_norm:.4f}):")
    print(f"  Normal correctly classified:   {normal_correct}/{len(normal_scores_normalized)} ({100*normal_correct/len(normal_scores_normalized):.1f}%)")
    print(f"  Abnormal correctly classified: {abnormal_correct}/{len(abnormal_scores_normalized)} ({100*abnormal_correct/len(abnormal_scores_normalized):.1f}%)")
    print(f"  Overall Accuracy: {100*accuracy:.1f}%")
    
    # Detailed breakdown on first sample from each class
    print("\n" + "=" * 80)
    print("DETAILED METRICS BREAKDOWN (First Sample from Each Class):")
    print("=" * 80)
    
    print("\nNormal Sample 0:")
    sig_normal = np.load(normal_files[0])
    metrics_normal_raw = compute_st_vae_metrics_breakdown(model, sig_normal, normalized=False)
    metrics_normal_norm = compute_st_vae_metrics_breakdown(model, sig_normal, normalized=True)
    print(f"  File: {normal_files[0].name}")
    print(f"  MSE:          {metrics_normal_raw['mse']:.4f}")
    print(f"  Weighted MSE: {metrics_normal_raw['weighted_mse']:.4f}")
    print(f"  KL Divergence: {metrics_normal_raw['kl_divergence']:.2f}")
    print(f"  NLL:          {metrics_normal_raw['nll']:.2f}")
    print(f"  Total Score (raw):        {metrics_normal_raw['total_score']:.2f}")
    print(f"  Total Score (normalized): {metrics_normal_norm['total_score']:.4f}")
    print(f"  Beta:         {metrics_normal_raw['beta']}")
    
    print("\nAbnormal Sample 0:")
    sig_abnormal = np.load(abnormal_files[0])
    metrics_abnormal_raw = compute_st_vae_metrics_breakdown(model, sig_abnormal, normalized=False)
    metrics_abnormal_norm = compute_st_vae_metrics_breakdown(model, sig_abnormal, normalized=True)
    print(f"  File: {abnormal_files[0].name}")
    print(f"  MSE:          {metrics_abnormal_raw['mse']:.4f}")
    print(f"  Weighted MSE: {metrics_abnormal_raw['weighted_mse']:.4f}")
    print(f"  KL Divergence: {metrics_abnormal_raw['kl_divergence']:.2f}")
    print(f"  NLL:          {metrics_abnormal_raw['nll']:.2f}")
    print(f"  Total Score (raw):        {metrics_abnormal_raw['total_score']:.2f}")
    print(f"  Total Score (normalized): {metrics_abnormal_norm['total_score']:.4f}")
    print(f"  Beta:         {metrics_abnormal_raw['beta']}")
    
    print("\nComponent Ratios (Abnormal/Normal):")
    print(f"  MSE:          {metrics_abnormal_raw['mse'] / metrics_normal_raw['mse']:.3f}x")
    print(f"  Weighted MSE: {metrics_abnormal_raw['weighted_mse'] / metrics_normal_raw['weighted_mse']:.3f}x")
    print(f"  KL Divergence: {metrics_abnormal_raw['kl_divergence'] / metrics_normal_raw['kl_divergence']:.3f}x")
    print(f"  NLL:          {metrics_abnormal_raw['nll'] / metrics_normal_raw['nll']:.3f}x")
    print(f"  Total Score (raw):        {metrics_abnormal_raw['total_score'] / metrics_normal_raw['total_score']:.3f}x")
    print(f"  Total Score (normalized): {metrics_abnormal_norm['total_score'] / metrics_normal_norm['total_score']:.3f}x")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT:")
    print("=" * 80)
    if mean_ratio_norm > 1.2 and accuracy > 0.7:
        print("SUCCESS: ST-VAE anomaly detection is working correctly!")
        print(f"  - Clear discrimination (normalized ratio={mean_ratio_norm:.2f}x > 1.2)")
        print(f"  - Good accuracy ({100*accuracy:.1f}% > 70%)")
    elif mean_ratio_norm > 1.1:
        print("MARGINAL: ST-VAE shows some discrimination but could be better")
        print(f"  - Moderate discrimination (normalized ratio={mean_ratio_norm:.2f}x)")
        print(f"  - Accuracy: {100*accuracy:.1f}%")
    else:
        print("FAILED: ST-VAE is not discriminating well on this batch")
        print(f"  - Poor discrimination (normalized ratio={mean_ratio_norm:.2f}x < 1.1)")
        print(f"  - Accuracy: {100*accuracy:.1f}%")
    
    print("\nNOTE: For presentations/reports, use normalized scores (0-1 range).")
    print("      Internal detection logic uses raw scores for actual classification.")
    print("=" * 80)

if __name__ == "__main__":
    test_st_vae_batch()
