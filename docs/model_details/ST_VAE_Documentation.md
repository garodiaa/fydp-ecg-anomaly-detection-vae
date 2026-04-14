# Spectral-Temporal VAE (ST-VAE) - Complete Technical Documentation

**Model Type:** Dual-Domain Variational Autoencoder  
**Architecture:** Parallel Time + Frequency Encoders with ResNet Backbone  
**Application:** ECG Anomaly Detection (Multi-Domain Analysis)  
**Parameters:** ~8.2M

---

## 1. Architecture Overview

### 1.1 Design Philosophy

The Spectral-Temporal VAE (ST-VAE) is built on the principle that **ECG abnormalities manifest in both time and frequency domains**:

- **Time Domain (Morphology):** P-wave shape, QRS complex width/amplitude, T-wave inversion, ST-segment elevation
- **Frequency Domain (Rhythm):** Heart rate variability, arrhythmias, periodic abnormalities, spectral power distribution

**Key Innovation:** Dual parallel encoders that process both representations simultaneously, then fuse into a unified latent space.

**Clinical Motivation:**
- **Myocardial Infarction (MI):** Visible as ST-elevation (time) AND reduced HRV (frequency)
- **Atrial Fibrillation:** Irregular R-R intervals (time) AND absent P-wave peaks (frequency)
- **Ventricular Hypertrophy:** Increased QRS amplitude (time) AND power spectrum shift (frequency)

### 1.2 Full Architecture Diagram

```
Input: (Batch, 1000 timesteps, 12 leads)
         ↓
    Transpose to (B, 12, 1000) for processing
         ↓
   ┌────────────────────────────────────────────┐
   │         PARALLEL DUAL ENCODERS             │
   └────────────────────────────────────────────┘
         ↓                          ↓
   [TIME BRANCH]              [SPECTRAL BRANCH]
   (Morphology)               (Rhythm/Frequency)
         ↓                          ↓
   Input: (B, 12, 1000)       FFT: torch.fft.rfft(x, dim=-1)
         ↓                    Magnitude: abs(x_fft)
   ResNet1DBlock              Shape: (B, 12, 501)
   12 → 32, k=7, s=2               ↓
   (B, 32, 500)              Normalize per lead:
         ↓                    x_fft_norm = x_fft / max(x_fft)
   ResNet1DBlock                   ↓
   32 → 64, k=5, s=2         Log1p scaling: log(1 + x_fft_norm)
   (B, 64, 250)              (B, 12, 501)
         ↓                          ↓
   ResNet1DBlock              ResNet1DBlock
   64 → 128, k=5, s=2         12 → 32, k=5, s=2
   (B, 128, 125)              (B, 32, 251)
         ↓                          ↓
   ResNet1DBlock              ResNet1DBlock
   128 → 256, k=3, s=2        32 → 64, k=5, s=2
   (B, 256, 62)               (B, 64, 126)
         ↓                          ↓
   AdaptiveAvgPool1d(32)      ResNet1DBlock
   (B, 256, 32)               64 → 128, k=3, s=2
         ↓                    (B, 128, 63)
   Flatten                         ↓
   (B, 8192)                  AdaptiveAvgPool1d(32)
                              (B, 128, 32)
                                   ↓
                              Flatten
                              (B, 4096)
         ↓                          ↓
   ┌────────────────────────────────────────────┐
   │              FUSION LAYER                  │
   └────────────────────────────────────────────┘
         ↓
   Concatenate: [time_feat, freq_feat]
   (B, 12288)
         ↓
   Linear(12288 → 512)
   LayerNorm(512)
   GELU activation
   Dropout(0.2)
   (B, 512)
         ↓
   ┌────────────────────────────────────────────┐
   │          VARIATIONAL BOTTLENECK            │
   └────────────────────────────────────────────┘
         ↓
   fc_mu:     Linear(512 → 32)      → mu: (B, 32)
   fc_logvar: Linear(512 → 32)      → logvar: (B, 32)
         ↓
   Reparameterization: z = mu + eps * exp(0.5 * logvar)
   z: (B, 32) ← Latent representation
         ↓
   ┌────────────────────────────────────────────┐
   │              DECODER                       │
   └────────────────────────────────────────────┘
         ↓
   Linear(32 → 256 * 125)
   LayerNorm
   GELU activation
   Dropout(0.2)
   Reshape: (B, 256, 125)
         ↓
   ConvTranspose1D(256 → 128, k=4, s=2, p=1)
   (B, 128, 250)
   BatchNorm1D, GELU, Dropout
         ↓
   ConvTranspose1D(128 → 64, k=4, s=2, p=1)
   (B, 64, 500)
   BatchNorm1D, GELU, Dropout
         ↓
   ConvTranspose1D(64 → 32, k=4, s=2, p=1)
   (B, 32, 1000)
   BatchNorm1D, GELU, Dropout
         ↓
   Conv1D(32 → 12, k=3, p=1)
   (B, 12, 1000)
         ↓
   Upsample if needed (to exactly 1000 samples)
         ↓
   Transpose to (B, 1000, 12)
         ↓
   Output: Reconstructed ECG (B, 1000, 12)
```

### 1.3 ResNet1D Block Details

```python
class ResNet1DBlock(nn.Module):
    """Residual block with skip connection"""
    
    Conv1D(in, out, k, s, dilation) + BatchNorm + GELU
         ↓
    Conv1D(out, out, k, 1, dilation) + BatchNorm
         ↓
    Add skip connection:
      if stride != 1 or in != out:
          shortcut = Conv1D(in, out, k=1, s) + BatchNorm
      else:
          shortcut = identity
         ↓
    out = GELU(conv_out + shortcut)
```

**Benefits:**
- ✅ **Gradient flow:** Skip connections prevent vanishing gradients
- ✅ **Deeper networks:** Enables 4-layer encoders without degradation
- ✅ **Feature reuse:** Identity mappings preserve low-level features

---

## 2. Forward Pass: Step-by-Step

### 2.1 Input Preparation

```python
Input x: (64, 1000, 12)

# Transpose to (B, C, T) for Conv1D
if x.dim() == 3 and x.size(1) != 12:
    x = x.transpose(1, 2)
x: (64, 12, 1000)
```

### 2.2 Time Branch (Morphology Encoding)

**Purpose:** Capture local ECG morphology (P-QRS-T wave shapes)

```python
# Layer 1: Initial feature extraction
time_feat = ResNet1DBlock(12 → 32, k=7, s=2)(x)
# (64, 32, 500) - Captures 14ms local patterns

# Layer 2: Mid-level features
time_feat = Dropout(0.2)(time_feat)
time_feat = ResNet1DBlock(32 → 64, k=5, s=2)(time_feat)
# (64, 64, 250) - Captures ~30ms QRS-like patterns

# Layer 3: High-level features
time_feat = Dropout(0.2)(time_feat)
time_feat = ResNet1DBlock(64 → 128, k=5, s=2)(time_feat)
# (64, 128, 125) - Captures beat-level morphology

# Layer 4: Abstract features
time_feat = Dropout(0.2)(time_feat)
time_feat = ResNet1DBlock(128 → 256, k=3, s=2)(time_feat)
# (64, 256, 62) - High-level abstractions

# Adaptive pooling to fixed size
time_feat = AdaptiveAvgPool1d(32)(time_feat)
# (64, 256, 32) - Consistent representation

# Flatten
time_feat = time_feat.flatten(1)
# (64, 8192) - Time domain features
```

### 2.3 Spectral Branch (Frequency Encoding)

**Purpose:** Capture rhythm, periodicity, and frequency-domain abnormalities

```python
# Step 1: FFT (Real-valued signals → Complex spectrum)
x_fft = torch.fft.rfft(x, dim=-1)  # Complex tensor
# x_fft: (64, 12, 501) - Positive frequencies only

# Step 2: Magnitude spectrum
x_fft_mag = x_fft.abs()
# (64, 12, 501) - Real-valued magnitudes

# Step 3: Normalize per lead (critical for stability)
x_fft_max = x_fft_mag.max(dim=-1, keepdim=True)[0] + 1e-6
x_fft_norm = x_fft_mag / x_fft_max
# (64, 12, 501) - Normalized to [0, 1]

# Step 4: Log1p scaling (compress dynamic range)
x_fft_log = torch.log1p(x_fft_norm)
# (64, 12, 501) - Smooth values in ~[0, 0.7]

# Frequency encoding (similar structure to time branch)
freq_feat = ResNet1DBlock(12 → 32, k=5, s=2)(x_fft_log)
# (64, 32, 251)
freq_feat = Dropout(0.2)(freq_feat)
freq_feat = ResNet1DBlock(32 → 64, k=5, s=2)(freq_feat)
# (64, 64, 126)
freq_feat = Dropout(0.2)(freq_feat)
freq_feat = ResNet1DBlock(64 → 128, k=3, s=2)(freq_feat)
# (64, 128, 63)
freq_feat = AdaptiveAvgPool1d(32)(freq_feat)
# (64, 128, 32)
freq_feat = freq_feat.flatten(1)
# (64, 4096) - Frequency domain features
```

**Why FFT Preprocessing?**
- **Normalization per lead:** Prevents large amplitude leads from dominating
- **Log1p scaling:** `log(1 + x)` compresses dynamic range without instability (no log(0))
- **Magnitude only:** Phase information discarded (less relevant for anomaly detection)

### 2.4 Fusion and Latent Encoding

```python
# Concatenate time and frequency features
combined = torch.cat([time_feat, freq_feat], dim=1)
# (64, 12288) - Unified multi-domain representation

# Fusion MLP
hidden = Linear(12288 → 512)(combined)
hidden = LayerNorm(512)(hidden)
hidden = GELU()(hidden)
hidden = Dropout(0.2)(hidden)
# (64, 512) - Compressed fusion

# Variational parameters
mu = Linear(512 → 32)(hidden)        # (64, 32)
logvar = Linear(512 → 32)(hidden)     # (64, 32)
logvar = torch.clamp(logvar, min=-10, max=10)

# Reparameterization
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + eps * std
# (64, 32) - Sampled latent code
```

### 2.5 Decoding (Time Domain Reconstruction)

```python
# Expand latent to spatial representation
d_in = Linear(32 → 256*125)(z)
d_in = LayerNorm(256*125)(d_in)
d_in = GELU()(d_in)
d_in = Dropout(0.2)(d_in)
d_in = d_in.view(64, 256, 125)
# (64, 256, 125) - Initial spatial features

# Transposed convolutions (upsampling)
recon = ConvTranspose1D(256 → 128, k=4, s=2, p=1)(d_in)
recon = BatchNorm1D(128)(recon)
recon = GELU()(recon)
recon = Dropout(0.2)(recon)
# (64, 128, 250)

recon = ConvTranspose1D(128 → 64, k=4, s=2, p=1)(recon)
recon = BatchNorm1D(64)(recon)
recon = GELU()(recon)
recon = Dropout(0.2)(recon)
# (64, 64, 500)

recon = ConvTranspose1D(64 → 32, k=4, s=2, p=1)(recon)
recon = BatchNorm1D(32)(recon)
recon = GELU()(recon)
recon = Dropout(0.2)(recon)
# (64, 32, 1000)

recon = Conv1D(32 → 12, k=3, p=1)(recon)
# (64, 12, 1000) - Final reconstruction

# Ensure exact length (transposed conv can be off by 1-2 samples)
if recon.size(-1) != 1000:
    recon = F.interpolate(recon, size=1000, mode='linear', align_corners=False)

# Transpose back to (B, T, C)
recon = recon.transpose(1, 2)
# (64, 1000, 12) - Reconstructed ECG
```

---

## 3. Loss Function (Multi-Domain)

### 3.1 Total Loss Formulation

```
Total Loss = L_time + α * (L_freq + 0.1 * L_spectral) + β * L_KL

Where:
  L_time      = Time domain reconstruction (MSE)
  L_freq      = Frequency domain reconstruction (L1)
  L_spectral  = Low-frequency emphasis (MSE on 0-50 Hz)
  α           = Learnable frequency weight (sigmoid-bounded)
  β           = KL divergence weight (0.5)
```

### 3.2 Time Domain Loss

```python
# MSE between original and reconstructed ECG
x = x.transpose(1, 2) if x.size(1) != 12 else x
x_mean = x_mean.transpose(1, 2) if x_mean.size(1) != 12 else x_mean

time_loss = F.mse_loss(x_mean, x, reduction='sum') / x.size(0)
```

**Typical Values:** 3,000 - 4,000 (normalized ECG)

### 3.3 Frequency Domain Loss

```python
# Original FFT
x_fft_orig = torch.fft.rfft(x, dim=-1).abs()
x_fft_orig_norm = x_fft_orig / (x_fft_orig.max(dim=-1, keepdim=True)[0] + 1e-6)
x_fft_orig_log = torch.log1p(x_fft_orig_norm)

# Reconstructed FFT
recon_fft = torch.fft.rfft(x_mean, dim=-1).abs()
recon_fft_norm = recon_fft / (recon_fft.max(dim=-1, keepdim=True)[0] + 1e-6)
recon_fft_log = torch.log1p(recon_fft_norm)

# L1 loss (more robust to outliers than MSE)
freq_loss = F.l1_loss(recon_fft_log, x_fft_orig_log, reduction='sum') / x.size(0)
```

**Why L1?**
- FFT magnitudes have outliers (sharp peaks at heart rate frequencies)
- L1 is less sensitive to large errors than MSE
- Provides better gradient signal

**Typical Values:** 90 - 160

### 3.4 Spectral Regularization (Low-Frequency Emphasis)

```python
# Focus on 0-50 Hz (clinically relevant for ECG)
freq_bins = x_fft_orig.size(-1)
low_freq_cutoff = min(50, freq_bins // 4)  # ~125 bins for 500 Hz sampling

spectral_reg = F.mse_loss(
    recon_fft[..., :low_freq_cutoff],
    x_fft_orig[..., :low_freq_cutoff],
    reduction='sum'
) / x.size(0)
```

**Why Low Frequencies?**
- ECG information concentrated below 50 Hz
- P-wave (0.5-5 Hz), QRS (10-40 Hz), T-wave (2-7 Hz)
- Higher frequencies often noise/muscle artifacts

**Typical Values:** 5 - 15

### 3.5 KL Divergence

```python
# Use first timestep (mu/logvar are expanded)
mu = mu[:, 0, :]       # (B, 32)
logvar = logvar[:, 0, :]  # (B, 32)

# KL with free bits (prevent collapse)
kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
kl_per_dim = torch.clamp(kl_per_dim, min=0.01)  # Free bits = 0.01
kl_loss = kl_per_dim.sum(dim=-1).mean()
```

**Free Bits:** Ensures each latent dimension carries at least 0.01 nats of information.

**Typical Values:** 120 - 140

### 3.6 Adaptive Frequency Weighting

```python
# Learnable parameter
self.freq_balance = nn.Parameter(torch.tensor(freq_weight))  # Init to 0.3

# Sigmoid-bounded (forces between 0 and 1)
freq_weight = torch.sigmoid(self.freq_balance)

# Total loss
total = time_loss + freq_weight * (freq_loss + 0.1 * spectral_reg) + beta * kl_loss
```

**Why Learnable?**
- Dataset-dependent balance between time/frequency importance
- Automatically adapts during training
- Converges to ~0.28 for PTB-XL

---

## 4. Performance Analysis

### 4.1 Evaluation Results (PTB-XL Test Set)

**Latest Results (from comprehensive JSON):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **F1-Score** | **0.90** | Excellent balance (best overall) |
| **Precision** | **0.89** | 89% of flagged cases truly abnormal |
| **Recall** | **0.91** | Catches 91% of all abnormalities |
| **AUC-ROC** | **0.93** | Excellent discrimination |
| **AUC-PR** | **0.91** | Strong on imbalanced data |
| **Threshold** | 0.35 | 95th percentile of normal errors |
| **Inference** | **0.108 ms/sample** | **Fastest VAE model** |

**Confusion Matrix:**
```
                  Predicted
                Normal  Abnormal
Actual Normal     8,133    1,381  (TN, FP)
Actual Abnormal   1,109   11,210  (FN, TP)
```

**Analysis:**
- **True Positive Rate (Recall):** 11,210 / 12,319 = **91.0%** ← Catches most abnormalities
- **True Negative Rate (Specificity):** 8,133 / 9,514 = **85.5%** ← Most normals correctly identified
- **False Positive Rate:** 1,381 / 9,514 = **14.5%** ← Some false alarms
- **False Negative Rate:** 1,109 / 12,319 = **9.0%** ← Few missed abnormalities

### 4.2 Comparison with Other Models

| Model | F1 | AUC-ROC | Precision | Recall | Inference (ms) | Params |
|-------|-----|---------|-----------|--------|----------------|--------|
| **ST-VAE** | **0.90** | **0.93** | 0.89 | **0.91** | **0.11** | 8.2M |
| BA-VAE | 0.55 | 0.78 | **0.91** | 0.40 | 1.53 | 0.6M |
| VAE-BiLSTM-Attn | 0.52 | 0.79 | 0.91 | 0.37 | 4.54 | 2.1M |
| HL-VAE | 0.45 | 0.75 | 0.89 | 0.30 | 3.72 | 2.0M |
| CAE | N/A | N/A | N/A | N/A | ~0.5 | 0.25M |

**ST-VAE Leads in:**
- ✅ **F1-Score** (+0.35 over BA-VAE)
- ✅ **Recall** (+0.51 over BA-VAE) ← Catches 2x more abnormalities!
- ✅ **AUC-ROC** (+0.15 over BA-VAE)
- ✅ **Inference Speed** (13x faster than VAE-BiLSTM-Attn)

**Why ST-VAE is Best:**
1. **Dual-domain learning:** Captures both morphology (time) AND rhythm (frequency)
2. **ResNet backbone:** Better gradient flow, deeper feature extraction
3. **Adaptive weighting:** Learns optimal time-freq balance
4. **No recurrence:** Parallel CNNs are faster than sequential RNNs
5. **Strong regularization:** β=0.5 prevents overfitting

### 4.3 Training Dynamics

**Key Training Milestones:**
```
Epoch 1-10:   Rapid loss decrease (learning basic patterns)
              MSE: 23,000 → 21,000
              
Epoch 10-20:  Learning rate reduces (5e-4 → 2.5e-4)
              MSE: 21,000 → 20,000
              
Epoch 20-40:  Refinement phase
              MSE: 20,000 → 15,000
              Learning rate → 1.25e-4
              
Epoch 40-50:  Final convergence
              MSE: 15,000 → 3,000
              Best epoch: 49 (val_loss = 5,932)
```

**Best Epoch Analysis:**
- **Epoch 49** (not early stopping!)
- Training loss: 6,509
- Validation loss: 5,932
- No overfitting (train ≈ val)
- Stable KL divergence: ~122-127

---

## 5. Why ST-VAE Performs Best

### 5.1 Dual-Domain Advantage

**Complementary Information:**

| Domain | What It Captures | Example Abnormality |
|--------|------------------|---------------------|
| **Time** | Morphology, amplitude, duration | ST-elevation (MI), QRS widening (bundle branch block) |
| **Frequency** | Rhythm, periodicity, power distribution | Atrial fibrillation (missing P-wave), VT (shifted spectrum) |

**Synergy:**
- Some abnormalities clearer in time (ST changes)
- Others clearer in frequency (irregular rhythms)
- **Fusion captures both** → Higher recall

### 5.2 Architectural Advantages

**ResNet Blocks:**
- Skip connections prevent vanishing gradients
- Enables 4-layer encoders (deeper than other models)
- Each layer learns residuals (what's different from identity)

**Adaptive Weighting:**
- Model learns `freq_weight ≈ 0.28` during training
- Not a fixed hyperparameter (adapts to data)
- Balances time vs frequency importance automatically

**Parallel Processing:**
- Time and frequency branches operate independently
- No sequential dependencies (unlike RNN/LSTM)
- Fully parallelizable on GPU → Fast inference

### 5.3 Loss Function Design

**Multi-Objective:**
- `L_time`: Ensures accurate time-domain reconstruction
- `L_freq`: Ensures spectral consistency
- `L_spectral`: Emphasizes clinically relevant low frequencies
- `L_KL`: Regularizes latent space

**Robust to Outliers:**
- L1 loss for frequency (less sensitive to FFT peaks)
- MSE for time (standard for waveform reconstruction)
- Free bits (prevents latent collapse)

### 5.4 Training Stability

**Key Improvements from Initial Version:**
- **Reduced latent_dim:** 64 → 32 (prevented KL explosion)
- **Increased beta:** 0.3 → 0.5 (stronger regularization)
- **Reduced freq_weight:** 0.5 → 0.3 (better balance)
- **Increased dropout:** 0.1 → 0.2 (reduced overfitting)

**Result:**
- Original ST-VAE: F1 = 0.066 (failed, epoch 3 best)
- Improved ST-VAE: F1 = 0.90 (success, epoch 49 best)
- **990% improvement!**

---

