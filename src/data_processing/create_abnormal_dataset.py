"""
Create abnormal ECG dataset from PTB-XL for anomaly detection evaluation.
Extracts ECGs with specific abnormal diagnostic codes (MI, HYP, STTC, CD, etc.)
and preprocesses them the same way as normal ECGs.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import wfdb
from scipy import signal
import ast
from tqdm import tqdm
import argparse

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # Go up to Dataset folder

def bandpass_filter(ecg_signal, lowcut=0.5, highcut=40, fs=100, order=2):
    """Apply bandpass filter to ECG signal"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, ecg_signal, axis=0)

def notch_filter(ecg_signal, freq=50.0, fs=100, quality=30):
    """Apply notch filter to remove powerline interference"""
    b, a = signal.iirnotch(freq, quality, fs)
    return signal.filtfilt(b, a, ecg_signal, axis=0)

def preprocess_ecg(ecg_data, fs=100):
    """Apply preprocessing pipeline to ECG data"""
    # Bandpass filter (0.5-40 Hz)
    ecg_filtered = bandpass_filter(ecg_data, lowcut=0.5, highcut=40, fs=fs)
    
    # Notch filter (50 Hz)
    ecg_filtered = notch_filter(ecg_filtered, freq=50.0, fs=fs)
    
    # Z-score normalization per lead
    mean = np.mean(ecg_filtered, axis=0, keepdims=True)
    std = np.std(ecg_filtered, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0  # Avoid division by zero
    ecg_normalized = (ecg_filtered - mean) / std
    
    return ecg_normalized

def main():
    parser = argparse.ArgumentParser(description='Create abnormal ECG dataset from PTB-XL')
    parser.add_argument('--input-dir', type=str, default='ptbxl', help='PTB-XL raw data directory')
    parser.add_argument('--output-dir', type=str, default='data/processed/ptbxl_abnormal', help='Output directory')
    parser.add_argument('--resolution', type=str, choices=['auto', 'hr', 'lr'], default='auto', 
                       help='Resolution: auto (try HR first), hr (500Hz), or lr (100Hz)')
    parser.add_argument('--abnormal-classes', nargs='+', 
                       default=['MI', 'HYP', 'STTC', 'CD'],
                       help='Diagnostic superclasses to include (MI, HYP, STTC, CD, NORM)')
    args = parser.parse_args()
    
    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PTB-XL database
    csv_path = input_dir / "ptbxl_database.csv"
    print(f"Loading database from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Load SCP statements for diagnostic class mapping
    scp_path = input_dir / "scp_statements.csv"
    scp_df = pd.read_csv(scp_path, index_col=0)
    
    # Create diagnostic class mapping
    diagnostic_class_map = scp_df['diagnostic_class'].to_dict()
    
    # Filter for abnormal ECGs with specified diagnostic classes
    def has_abnormal_class(scp_codes_str):
        try:
            codes = ast.literal_eval(scp_codes_str)
            for code in codes.keys():
                if code in diagnostic_class_map:
                    diag_class = diagnostic_class_map[code]
                    if diag_class in args.abnormal_classes:
                        return True
        except:
            return False
        return False
    
    print(f"\\nFiltering for abnormal ECGs with diagnostic classes: {args.abnormal_classes}")
    abnormal_mask = df['scp_codes'].apply(has_abnormal_class)
    abnormal_df = df[abnormal_mask].copy()
    
    print(f"Found {len(abnormal_df)} abnormal ECG records")
    print(f"\\nDiagnostic class distribution:")
    
    # Count samples per diagnostic class
    class_counts = {}
    for scp_codes_str in abnormal_df['scp_codes']:
        codes = ast.literal_eval(scp_codes_str)
        for code in codes.keys():
            if code in diagnostic_class_map:
                diag_class = diagnostic_class_map[code]
                if diag_class in args.abnormal_classes:
                    class_counts[diag_class] = class_counts.get(diag_class, 0) + 1
    
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    
    # Process each abnormal ECG
    processed_count = 0
    skipped_count = 0
    
    print(f"\\nProcessing abnormal ECGs...")
    for idx, row in tqdm(abnormal_df.iterrows(), total=len(abnormal_df), desc="Processing"):
        ecg_id = row['ecg_id']
        
        # Determine file path based on resolution
        if args.resolution == 'hr':
            rec_name = row['filename_hr']
        elif args.resolution == 'lr':
            rec_name = row['filename_lr']
        else:  # auto
            hr_name = row['filename_hr']
            lr_name = row['filename_lr']
            
            # Try HR first
            hr_path = input_dir / f"{hr_name}.hea"
            if hr_path.exists():
                rec_name = hr_name
            else:
                rec_name = lr_name
        
        rec_path = input_dir / rec_name
        
        # Check if .hea file exists
        hea_file = Path(str(rec_path) + '.hea')
        if not hea_file.exists():
            skipped_count += 1
            continue
        
        try:
            # Read ECG using WFDB
            record = wfdb.rdrecord(str(rec_path))
            ecg_data = record.p_signal  # Shape: (n_samples, 12)
            
            # Skip if not 12-lead
            if ecg_data.shape[1] != 12:
                skipped_count += 1
                continue
            
            # Preprocess
            ecg_processed = preprocess_ecg(ecg_data, fs=record.fs)
            
            # Save as .npy
            output_file = output_dir / f"{ecg_id:05d}.npy"
            np.save(output_file, ecg_processed)
            processed_count += 1
            
        except Exception as e:
            print(f"\\nError processing ECG {ecg_id}: {e}")
            skipped_count += 1
            continue
    
    print(f"\\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Successfully processed: {processed_count} abnormal ECGs")
    print(f"Skipped: {skipped_count} ECGs")
    print(f"Output directory: {output_dir}")
    
    # Save metadata CSV
    metadata_path = output_dir / "abnormal_metadata.csv"
    abnormal_df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to: {metadata_path}")
    
    # Save diagnostic class summary
    summary_path = output_dir / "diagnostic_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Abnormal ECG Dataset Summary\\n")
        f.write(f"{'='*60}\\n\\n")
        f.write(f"Total abnormal ECGs: {processed_count}\\n")
        f.write(f"Diagnostic classes included: {', '.join(args.abnormal_classes)}\\n\\n")
        f.write(f"Distribution by diagnostic class:\\n")
        for cls, count in sorted(class_counts.items()):
            f.write(f"  {cls}: {count}\\n")
        f.write(f"\\n{'='*60}\\n")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
