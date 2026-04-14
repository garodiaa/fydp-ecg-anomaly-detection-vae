# ECG Anomaly Detection Using Advanced VAE Architectures

![Version](https://img.shields.io/badge/version-1.0.0-283593?style=flat)
![Project](https://img.shields.io/badge/Project-Final%20Year%20Thesis-1565C0?style=flat)
![Task](https://img.shields.io/badge/Task-ECG%20Anomaly%20Detection-00897B?style=flat)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=flat)
![Dataset](https://img.shields.io/badge/Dataset-PTB--XL-2E7D32?style=flat)
![Models](https://img.shields.io/badge/Models-6%20VAE%20Variants-6D4C41?style=flat)
![License](https://img.shields.io/badge/License-Academic-4CAF50?style=flat)

Final year project focused on deep-learning based ECG anomaly detection using multiple Variational Autoencoder (VAE) families. This repository contains the full experimentation pipeline, model implementations, evaluation suite, and thesis-grade documentation.

## Project Highlights

- Comprehensive comparison of VAE architectures on PTB-XL ECG data.
- End-to-end pipeline: preprocessing, training, evaluation, and visualization.
- Strong reported results with ST-VAE as top-performing model.
- Structured for reproducibility and academic presentation.

## Results Snapshot

Based on the consolidated report in `docs/reports/model_report.md`:

| Model | F1-Score | AUC-ROC | Notes |
|------|---------:|--------:|-------|
| ST-VAE | 0.9000 | 0.9300 | Best overall performance |
| BA-VAE | 0.8995 | 0.9300 | Strong precision and stability |
| HL-VAE | 0.8589 | 0.8800 | Best precision-recall balance among hierarchical methods |

## Repository Structure

```text
Dataset/
	assets/
		ui/                    # UI image assets
	docs/
		guides/                # Methodology and scoring guides
		model_details/         # Per-model technical documentation
		reports/               # Final report and study material
	notebooks/               # Analysis and paper visualization notebooks
	outputs/                 # Generated experiment outputs
	src/
		anomaly/               # Anomaly scoring utilities
		data_processing/       # Data loading and preprocessing
		models/                # Model architectures
		training.py            # Main training entry point
		evaluate.py            # Evaluation pipeline
	tests/                   # Validation and smoke tests
```

## Tech Stack

- Python, PyTorch, NumPy, pandas, scikit-learn
- Jupyter for analysis workflows
- CUDA-enabled training support (optional)
- PTB-XL ECG dataset

## Quick Start

1. Install dependencies

```powershell
pip install -r requirements.txt
```

2. Optional: verify GPU

```powershell
python tests/test_gpu.py
```

3. Train a model (example: ST-VAE)

```powershell
python src/training.py --model st_vae --epochs 50
```

4. Run anomaly-detection sanity test

```powershell
python tests/test_anomaly_detection.py
```

## Reproducibility and Evaluation

- Training and metrics scripts are available in `src/`.
- Comparative visualizations are under `visualizations/` and `outputs/`.
- Detailed study methodology and analysis are documented in `docs/reports/study.md`.

## Documentation

- Main documentation index: `docs/README.md`
- Full report: `docs/reports/model_report.md`
- Hyperparameter comparison: `docs/reports/HYPERPARAMETERS_COMPARISON.md`
- Normalized scoring guide: `docs/guides/NORMALIZED_SCORING_GUIDE.md`

## Academic Context

This work was developed as a final year project and is intended for academic review, portfolio presentation, and further research applications.