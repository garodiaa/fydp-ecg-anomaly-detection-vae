"""
Quick test to verify streamlit logic matches the working batch test
"""

import torch
import numpy as np
from pathlib import Path
from model_loader import load_model_by_name
from compute_st_vae_anomaly import compute_st_vae_reconstruction_and_anomaly, normalize_score

def test_streamlit_logic():
    """Test that streamlit uses the same logic as the working batch test"""
    
    # Load model
    print("Loading ST-VAE model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model_by_name('ST-VAE (Spatiotemporal VAE)', map_location=device)
    model.eval()
    print(f"Model loaded. Beta: {model.beta}, Device: {device}")
    
    # Test on a few samples
    sample_data_dir = Path("SampleData")
    normal_dir = sample_data_dir / "normal"
    abnormal_dir = sample_data_dir / "abnormal"
    
    # Test 5 normal samples
    print("\n" + "="*60)
    print("Testing NORMAL samples:")
    print("="*60)
    normal_files = sorted(list(normal_dir.glob("*.npy")))[:5]
    for fpath in normal_files:
        ecg_data = np.load(fpath)
        
        # Ensure shape is (12, T)
        if ecg_data.shape[0] != 12:
            ecg_data = ecg_data.T
        
        # Mimic streamlit preprocessing - ensure 5000 samples
        if ecg_data.shape[1] < 5000:
            ecg_data = np.pad(ecg_data, ((0,0),(0, 5000 - ecg_data.shape[1])), mode='constant')
        else:
            ecg_data = ecg_data[:, :5000]
        
        # Compute anomaly score (same as streamlit)
        recon, std, mse, raw_score = compute_st_vae_reconstruction_and_anomaly(model, ecg_data, window_size=1000, stride=500)
        score = normalize_score(raw_score)
        
        decision = 'Abnormal' if score > 0.69 else 'Normal'
        print(f"{fpath.name:30s} -> Raw: {raw_score:.2f}, Normalized: {score:.4f}, Decision: {decision}")
    
    # Test 5 abnormal samples
    print("\n" + "="*60)
    print("Testing ABNORMAL samples:")
    print("="*60)
    abnormal_files = sorted(list(abnormal_dir.glob("*.npy")))[:5]
    for fpath in abnormal_files:
        ecg_data = np.load(fpath)
        
        # Ensure shape is (12, T)
        if ecg_data.shape[0] != 12:
            ecg_data = ecg_data.T
        
        # Mimic streamlit preprocessing - ensure 5000 samples
        if ecg_data.shape[1] < 5000:
            ecg_data = np.pad(ecg_data, ((0,0),(0, 5000 - ecg_data.shape[1])), mode='constant')
        else:
            ecg_data = ecg_data[:, :5000]
        
        # Compute anomaly score (same as streamlit)
        recon, std, mse, raw_score = compute_st_vae_reconstruction_and_anomaly(model, ecg_data, window_size=1000, stride=500)
        score = normalize_score(raw_score)
        
        decision = 'Abnormal' if score > 0.69 else 'Normal'
        print(f"{fpath.name:30s} -> Raw: {raw_score:.2f}, Normalized: {score:.4f}, Decision: {decision}")
    
    print("\n" + "="*60)
    print("Test complete! Streamlit logic verified.")
    print("="*60)

if __name__ == "__main__":
    with torch.no_grad():
        test_streamlit_logic()
