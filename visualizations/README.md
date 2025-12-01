# Model Visualizations

This directory contains comprehensive visualizations for all trained ECG anomaly detection models.

## Directory Structure

```
visualizations/
├── README.md (this file)
├── [Model-specific folders]
│   ├── ba_vae/
│   ├── cae/
│   ├── hlvae/
│   ├── st_vae/
│   ├── vae_bilstm_attn/
│   ├── vae_bilstm_mha/
│   └── vae_gru/
└── [Cross-model comparison visualizations]
```

## Model-Specific Visualizations

Each model folder contains the following visualizations:

### 1. **training_curves.png**
- **Description**: Comprehensive view of training progress across all epochs
- **Subplots**:
  - **Total Loss**: Combined loss (reconstruction + KL divergence for VAEs)
  - **Reconstruction Loss**: MSE or similar reconstruction error
  - **KL Divergence**: Regularization term (VAE models only)
  - **Learning Rate**: Schedule showing adaptive learning rate changes
- **Key Features**:
  - Train vs. validation comparison
  - Best epoch marked with vertical line
  - Helps identify overfitting and convergence patterns

### 2. **confusion_matrix.png**
- **Description**: Normalized confusion matrix with evaluation metrics
- **Elements**:
  - True Positives (TP), True Negatives (TN)
  - False Positives (FP), False Negatives (FN)
  - Annotated with actual counts
- **Metrics Box**:
  - Precision, Recall, F1-Score, AUC-ROC
- **Use Case**: Understanding classification errors and model reliability

### 3. **roc_curve.png**
- **Description**: Receiver Operating Characteristic curve
- **Elements**:
  - True Positive Rate vs. False Positive Rate
  - Area Under Curve (AUC-ROC) score
  - Operating point from confusion matrix
  - Random classifier baseline (diagonal line)
- **Interpretation**: Higher AUC indicates better discrimination ability

### 4. **pr_curve.png**
- **Description**: Precision-Recall curve
- **Elements**:
  - Precision vs. Recall tradeoff
  - Area Under Curve (AUC-PR) score
  - Current operating point
  - Random classifier baseline
- **Use Case**: Important for imbalanced datasets, focuses on positive class performance

### 5. **performance_summary.png**
- **Description**: Comprehensive 4-panel performance overview
- **Panels**:
  1. **Classification Metrics**: Bar chart of all key metrics
  2. **Confusion Matrix Distribution**: Pie chart breakdown
  3. **Model Architecture Info**: Parameters, hyperparameters, latency
  4. **Training Summary**: Dataset info, epoch details, test results
- **Use Case**: Quick reference for all model information

### 6. **loss_components.png**
- **Description**: Detailed breakdown of loss terms during training
- **Subplots**:
  - **Training Loss Components**: Total, Reconstruction, KL Divergence
  - **Validation Loss Components**: Same metrics on validation set
- **Use Case**: Understanding contribution of each loss term to model optimization

## Cross-Model Comparison Visualizations

### 1. **model_comparison_metrics.png**
- **Description**: Side-by-side comparison of key performance metrics
- **Panels**:
  - **F1-Score**: Overall classification performance
  - **AUC-ROC**: Discrimination ability
  - **Precision vs Recall**: Scatter plot showing tradeoff
  - **Inference Time**: Computational efficiency
- **Use Case**: Identifying best overall performer and performance tradeoffs

### 2. **training_convergence_comparison.png**
- **Description**: Training and validation loss curves for all models
- **Features**:
  - Overlaid curves for easy comparison
  - Log scale for better visibility across different loss magnitudes
  - Shows which models converge faster/better
- **Use Case**: Understanding training dynamics and stability

### 3. **architecture_comparison.png**
- **Description**: Model complexity and efficiency analysis
- **Panels**:
  - **Model Size**: Total parameters (in millions)
  - **Model Efficiency**: F1-Score per million parameters
- **Use Case**: Evaluating model complexity vs. performance tradeoff

### 4. **radar_comparison.png**
- **Description**: Multi-dimensional performance comparison
- **Dimensions**:
  - F1-Score
  - AUC-ROC
  - Precision
  - Recall
  - Speed (inverse of latency)
- **Use Case**: Holistic view of model strengths and weaknesses

### 5. **summary_table.png**
- **Description**: Comprehensive table with all key metrics
- **Columns**:
  - Model name
  - Classification metrics (F1, Precision, Recall, AUC-ROC, AUC-PR)
  - Architecture info (Parameters)
  - Performance (Latency)
  - Training info (Best epoch / Total epochs)
- **Features**: Best values highlighted in green
- **Use Case**: Quick reference for all model statistics

### 6. **confusion_matrices_comparison.png**
- **Description**: Grid view of normalized confusion matrices for all models
- **Features**:
  - Normalized by true labels (percentages)
  - F1-Score displayed in title
  - Side-by-side comparison of error patterns
- **Use Case**: Understanding how different models make errors

## Models Included

1. **BA-VAE (Beat-Aligned VAE)**: Beat-aligned variational autoencoder with per-beat latent representation
2. **CAE (Convolutional Autoencoder)**: Baseline convolutional autoencoder
3. **HLVAE (Hierarchical Latent VAE)**: VAE with global and local latent representations
4. **ST-VAE**: Spatial-temporal VAE for ECG modeling
5. **VAE BiLSTM Attn**: BiLSTM encoder-decoder with lead-wise and temporal attention
6. **VAE BiLSTM MHA**: BiLSTM with multi-head attention mechanism
7. **VAE GRU**: GRU-based variational autoencoder

## Key Findings Summary

### Best Performers (by metric)

- **Highest F1-Score**: ST-VAE (0.900)
- **Highest AUC-ROC**: ST-VAE (0.930)
- **Highest Precision**: ST-VAE (0.890)
- **Highest Recall**: ST-VAE (0.910)
- **Fastest Inference**: CAE (0.019 ms/sample)
- **Most Parameters**: HLVAE (10.48M)
- **Most Efficient**: CAE (9.15 F1/M params)

### Model Characteristics

- **ST-VAE**: Best overall performance with excellent balance of precision and recall
- **VAE BiLSTM MHA**: Strong performance with moderate complexity
- **BA-VAE**: Good performance with beat-aligned representation
- **HLVAE**: High complexity but solid results with hierarchical structure
- **VAE BiLSTM Attn**: Competitive performance with attention mechanisms
- **VAE GRU**: Lightweight with reasonable performance
- **CAE**: Fastest and simplest baseline, lower performance but very efficient

## Generating Visualizations

To regenerate all visualizations:

```bash
# Individual model visualizations
python generate_visualizations.py

# Cross-model comparisons
python generate_model_comparisons.py
```

## Data Source

All visualizations are generated from the `*_comprehensive.json` files located in each model's output directory:
- `outputs/{model_id}/{model_id}_comprehensive.json`

## Notes

- All plots use consistent color schemes and styling for easy comparison
- High-resolution PNG format (300 DPI) suitable for publications
- Confusion matrices show both raw counts and percentages
- ROC and PR curves include operating points from actual model predictions
- Training curves mark the best performing epoch
- All metrics are calculated on the test set unless otherwise specified

---

**Generated**: December 2025  
**Dataset**: PTB-XL ECG Database  
**Task**: ECG Anomaly Detection (Normal vs Abnormal Classification)
