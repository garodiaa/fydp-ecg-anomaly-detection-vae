"""
Find 100 normal and 100 abnormal samples that ST-VAE correctly classifies.
Copy them to a demonstration folder for practical use.
"""

import torch
import numpy as np
import os
from pathlib import Path
import shutil

# Import the model loading function
import sys
sys.path.append(str(Path(__file__).parent / 'src'))
from generate_reconstructions import load_model

# Import our ST-VAE anomaly computation function
from compute_st_vae_anomaly import compute_st_vae_anomaly_simple

def find_correctly_classified_samples():
    """Find samples that ST-VAE correctly classifies"""
    
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
    
    # Get all sample files
    normal_files = sorted(list(normal_dir.glob("*.npy")))
    abnormal_files = sorted(list(abnormal_dir.glob("*.npy")))
    
    print(f"Available: {len(normal_files)} normal, {len(abnormal_files)} abnormal samples")
    print("=" * 80)
    
    # First pass: compute all scores to find threshold
    print("\nPhase 1: Computing scores for all samples...")
    print("-" * 80)
    
    normal_scores = []
    abnormal_scores = []
    
    print(f"Processing {len(normal_files)} normal samples...")
    for i, file_path in enumerate(normal_files):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(normal_files)}")
        sig_np = np.load(file_path)
        score = compute_st_vae_anomaly_simple(model, sig_np)
        normal_scores.append((file_path, score))
    
    print(f"Processing {len(abnormal_files)} abnormal samples...")
    for i, file_path in enumerate(abnormal_files):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(abnormal_files)}")
        sig_np = np.load(file_path)
        score = compute_st_vae_anomaly_simple(model, sig_np)
        abnormal_scores.append((file_path, score))
    
    # Extract just the scores for analysis
    normal_scores_only = np.array([s[1] for s in normal_scores])
    abnormal_scores_only = np.array([s[1] for s in abnormal_scores])
    
    print(f"\nCompleted scoring:")
    print(f"  Normal: min={normal_scores_only.min():.2f}, max={normal_scores_only.max():.2f}, mean={normal_scores_only.mean():.2f}")
    print(f"  Abnormal: min={abnormal_scores_only.min():.2f}, max={abnormal_scores_only.max():.2f}, mean={abnormal_scores_only.mean():.2f}")
    print("=" * 80)
    
    # Find optimal threshold using different strategies
    print("\nPhase 2: Finding optimal threshold...")
    print("-" * 80)
    
    # Strategy 1: Mean of means
    threshold_mean = (normal_scores_only.mean() + abnormal_scores_only.mean()) / 2
    
    # Strategy 2: Median of medians
    threshold_median = (np.median(normal_scores_only) + np.median(abnormal_scores_only)) / 2
    
    # Strategy 3: Minimize error rate
    thresholds_to_test = np.linspace(
        min(normal_scores_only.min(), abnormal_scores_only.min()),
        max(normal_scores_only.max(), abnormal_scores_only.max()),
        1000
    )
    
    best_threshold = None
    best_accuracy = 0
    best_counts = None
    
    for thresh in thresholds_to_test:
        normal_correct = np.sum(normal_scores_only < thresh)
        abnormal_correct = np.sum(abnormal_scores_only >= thresh)
        accuracy = (normal_correct + abnormal_correct) / (len(normal_scores_only) + len(abnormal_scores_only))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh
            best_counts = (normal_correct, abnormal_correct)
    
    print(f"Threshold options:")
    print(f"  Mean-based: {threshold_mean:.2f}")
    print(f"  Median-based: {threshold_median:.2f}")
    print(f"  Optimal (max accuracy): {best_threshold:.2f} -> {best_accuracy*100:.2f}% accuracy")
    print(f"    Normal correct: {best_counts[0]}/{len(normal_scores_only)} ({100*best_counts[0]/len(normal_scores_only):.1f}%)")
    print(f"    Abnormal correct: {best_counts[1]}/{len(abnormal_scores_only)} ({100*best_counts[1]/len(abnormal_scores_only):.1f}%)")
    
    # Use optimal threshold
    threshold = best_threshold
    print(f"\nUsing threshold: {threshold:.2f}")
    print("=" * 80)
    
    # Phase 3: Select correctly classified samples
    print("\nPhase 3: Selecting correctly classified samples...")
    print("-" * 80)
    
    # Normal samples: score < threshold
    correctly_classified_normal = [(f, s) for f, s in normal_scores if s < threshold]
    correctly_classified_normal.sort(key=lambda x: x[1])  # Sort by score (lowest first)
    
    # Abnormal samples: score >= threshold
    correctly_classified_abnormal = [(f, s) for f, s in abnormal_scores if s >= threshold]
    correctly_classified_abnormal.sort(key=lambda x: x[1], reverse=True)  # Sort by score (highest first)
    
    print(f"Found {len(correctly_classified_normal)} correctly classified normal samples")
    print(f"Found {len(correctly_classified_abnormal)} correctly classified abnormal samples")
    
    # Select top 100 from each (if available)
    n_normal = min(100, len(correctly_classified_normal))
    n_abnormal = min(100, len(correctly_classified_abnormal))
    
    selected_normal = correctly_classified_normal[:n_normal]
    selected_abnormal = correctly_classified_abnormal[:n_abnormal]
    
    print(f"\nSelecting {n_normal} normal and {n_abnormal} abnormal samples")
    print("=" * 80)
    
    # Phase 4: Copy to demonstration folder
    print("\nPhase 4: Copying files to demonstration folder...")
    print("-" * 80)
    
    demo_dir = Path("data/demo_samples")
    demo_normal_dir = demo_dir / "normal"
    demo_abnormal_dir = demo_dir / "abnormal"
    
    # Create directories
    demo_normal_dir.mkdir(parents=True, exist_ok=True)
    demo_abnormal_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy normal samples
    print(f"Copying {n_normal} normal samples...")
    for i, (file_path, score) in enumerate(selected_normal):
        dest = demo_normal_dir / file_path.name
        shutil.copy2(file_path, dest)
        if i < 5:
            print(f"  {file_path.name} -> score: {score:.2f}")
    if n_normal > 5:
        print(f"  ... and {n_normal - 5} more")
    
    # Copy abnormal samples
    print(f"Copying {n_abnormal} abnormal samples...")
    for i, (file_path, score) in enumerate(selected_abnormal):
        dest = demo_abnormal_dir / file_path.name
        shutil.copy2(file_path, dest)
        if i < 5:
            print(f"  {file_path.name} -> score: {score:.2f}")
    if n_abnormal > 5:
        print(f"  ... and {n_abnormal - 5} more")
    
    print("=" * 80)
    
    # Create metadata file
    metadata_file = demo_dir / "README.txt"
    with open(metadata_file, 'w') as f:
        f.write("ST-VAE Correctly Classified Demonstration Samples\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: ST-VAE (Beta: {model.beta})\n")
        f.write(f"Threshold: {threshold:.2f}\n")
        f.write(f"Overall Accuracy on Full Dataset: {best_accuracy*100:.2f}%\n\n")
        f.write(f"Normal Samples: {n_normal}\n")
        f.write(f"  Score Range: {selected_normal[0][1]:.2f} - {selected_normal[-1][1]:.2f}\n")
        f.write(f"  All scores < {threshold:.2f} (correctly classified as normal)\n\n")
        f.write(f"Abnormal Samples: {n_abnormal}\n")
        f.write(f"  Score Range: {selected_abnormal[-1][1]:.2f} - {selected_abnormal[0][1]:.2f}\n")
        f.write(f"  All scores >= {threshold:.2f} (correctly classified as abnormal)\n\n")
        f.write("Usage:\n")
        f.write("  - Normal samples are in normal/ subdirectory\n")
        f.write("  - Abnormal samples are in abnormal/ subdirectory\n")
        f.write("  - All samples are guaranteed to be correctly classified by ST-VAE\n")
        f.write("  - Use for practical demonstrations, visualizations, or further analysis\n\n")
        f.write("File Shapes:\n")
        f.write("  - Normal files: [12, 5000] (12 leads, 5000 samples)\n")
        f.write("  - Abnormal files: [5000, 12] or [12, 5000] (varies)\n")
        f.write("  - compute_st_vae_anomaly.py handles both formats automatically\n")
    
    print(f"\nSUMMARY:")
    print(f"  Created demonstration dataset: {demo_dir}")
    print(f"  Normal samples: {n_normal} files in {demo_normal_dir}")
    print(f"  Abnormal samples: {n_abnormal} files in {demo_abnormal_dir}")
    print(f"  Threshold used: {threshold:.2f}")
    print(f"  All samples are correctly classified by ST-VAE")
    print(f"  Metadata saved to: {metadata_file}")
    print("=" * 80)
    
    # Show statistics
    print("\nDEMO DATASET STATISTICS:")
    print(f"Normal scores:")
    print(f"  Min: {selected_normal[0][1]:.2f}")
    print(f"  Max: {selected_normal[-1][1]:.2f}")
    print(f"  Mean: {np.mean([s[1] for s in selected_normal]):.2f}")
    print(f"  Std: {np.std([s[1] for s in selected_normal]):.2f}")
    
    print(f"\nAbnormal scores:")
    print(f"  Min: {selected_abnormal[-1][1]:.2f}")
    print(f"  Max: {selected_abnormal[0][1]:.2f}")
    print(f"  Mean: {np.mean([s[1] for s in selected_abnormal]):.2f}")
    print(f"  Std: {np.std([s[1] for s in selected_abnormal]):.2f}")
    
    print(f"\nSeparation:")
    print(f"  Gap between classes: {selected_abnormal[-1][1] - selected_normal[-1][1]:.2f}")
    print(f"  Ratio (abnormal/normal means): {np.mean([s[1] for s in selected_abnormal]) / np.mean([s[1] for s in selected_normal]):.3f}x")
    print("=" * 80)

if __name__ == "__main__":
    find_correctly_classified_samples()
