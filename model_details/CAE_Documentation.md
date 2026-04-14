# Convolutional Autoencoder (CAE) - Complete Technical Documentation

**Model Type:** Convolutional Autoencoder (Non-Variational)  
**Architecture:** 1D CNN Encoder-Decoder  
**Application:** ECG Anomaly Detection (Baseline Model)  
**Parameters:** ~250K

---

## 1. Architecture Overview

### 1.1 Design Philosophy

The Convolutional Autoencoder (CAE) represents the **simplest baseline** for ECG anomaly detection. It uses 1D convolutions to:

- **Extract hierarchical features** through progressive downsampling
- **Compress ECG signals** into a learned bottleneck representation
- **Reconstruct signals** via transposed convolutions (deconvolution)
- **Detect anomalies** through reconstruction error (no probabilistic modeling)

**Key Differences from VAEs:**
- ❌ No latent space (no KL divergence)
- ❌ No probabilistic modeling (deterministic)
- ❌ No sampling/reparameterization
- ✅ Simpler training (only reconstruction loss)
- ✅ Faster inference
- ✅ Smaller model size

**Based on:** CAE-M architecture from TKDE 2021 paper on time-series anomaly detection.

### 1.2 Full Architecture Diagram

```
Input: (Batch, 12 leads, 1000 timesteps)  ← Transposed from (B, T, C)
         ↓
   [ENCODER - Progressive Downsampling]
         ↓
   Conv1D(12 → 32, kernel=7, stride=2, padding=3)
   Shape: (B, 32, 500)  ← Halved length
   ReLU activation
         ↓
   Conv1D(32 → 64, kernel=5, stride=2, padding=2)
   Shape: (B, 64, 250)  ← Halved again
   ReLU activation
         ↓
   Conv1D(64 → 128, kernel=3, stride=2, padding=1)
   Shape: (B, 128, 125)  ← Bottleneck (8x compression)
   ReLU activation
         ↓
   [BOTTLENECK]
   Encoded representation: (B, 128, 125)
   Compression ratio: (12×1000) / (128×125) = 12,000 / 16,000 = 0.75
   (Note: This is spatial compression, not parameter compression)
         ↓
   [DECODER - Progressive Upsampling]
         ↓
   ConvTranspose1D(128 → 64, kernel=3, stride=2, padding=1, output_padding=1)
   Shape: (B, 64, 250)  ← Doubled length
   ReLU activation
         ↓
   ConvTranspose1D(64 → 32, kernel=5, stride=2, padding=2, output_padding=1)
   Shape: (B, 32, 500)  ← Doubled again
   ReLU activation
         ↓
   ConvTranspose1D(32 → 12, kernel=7, stride=2, padding=3, output_padding=1)
   Shape: (B, 12, 1000)  ← Original dimensions restored
   (No activation - linear output)
         ↓
   Trim to exact length: [:, :, :1000]
         ↓
   Output: Reconstructed ECG (B, 12, 1000)
```

### 1.3 Key Architectural Features

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Input Format** | (B, 12, 1000) | Channels-first for Conv1D |
| **Encoder Layers** | 3 Conv1D layers | Progressive feature extraction |
| **Stride** | 2 at each layer | 8x total downsampling |
| **Kernel Sizes** | 7 → 5 → 3 | Large to small receptive fields |
| **Channels** | 12 → 32 → 64 → 128 | Increasing feature maps |
| **Activation** | ReLU | Simple, fast, non-saturating |
| **Decoder Type** | ConvTranspose1D (Deconvolution) | Learned upsampling |
| **Output Activation** | None (linear) | Direct reconstruction |

---

## 2. Forward Pass: Step-by-Step

### 2.1 Input Preprocessing

```python
# Training pipeline passes (B, T, C) format
x_input: (64, 1000, 12)

# Transpose to (B, C, T) for Conv1D
x = x_input.transpose(1, 2)
x: (64, 12, 1000)
```

### 2.2 Encoding Phase

**Layer 1: Initial Feature Extraction**
```
Conv1D(in=12, out=32, kernel=7, stride=2, padding=3)
Input:  (64, 12, 1000)
Output: (64, 32, 500)
Receptive field: 7 samples (14 ms at 500 Hz)
ReLU: element-wise activation
```

**Layer 2: Mid-Level Features**
```
Conv1D(in=32, out=64, kernel=5, stride=2, padding=2)
Input:  (64, 32, 500)
Output: (64, 64, 250)
Receptive field: 7 + 2*(5-1) = 15 samples (30 ms)
ReLU: element-wise activation
```

**Layer 3: High-Level Features (Bottleneck)**
```
Conv1D(in=64, out=128, kernel=3, stride=2, padding=1)
Input:  (64, 64, 250)
Output: (64, 128, 125)  ← Encoded representation
Receptive field: 15 + 4*(3-1) = 23 samples (46 ms)
ReLU: element-wise activation
```

**Bottleneck Analysis:**
- **Spatial compression:** 1000 → 125 (8x reduction)
- **Channel expansion:** 12 → 128 (10.7x increase)
- **Total elements:** 12×1000 = 12,000 → 128×125 = 16,000
  - *Slight expansion* (not true compression)
  - Information is **distributed** across more channels
  - Each channel captures specific features (edges, peaks, rhythms)

### 2.3 Decoding Phase

**Layer 1: Initial Upsampling**
```
ConvTranspose1D(in=128, out=64, kernel=3, stride=2, padding=1, output_padding=1)
Input:  (64, 128, 125)
Output: (64, 64, 250)
ReLU: element-wise activation
```

**Layer 2: Mid-Level Upsampling**
```
ConvTranspose1D(in=64, out=32, kernel=5, stride=2, padding=2, output_padding=1)
Input:  (64, 64, 250)
Output: (64, 32, 500)
ReLU: element-wise activation
```

**Layer 3: Final Reconstruction**
```
ConvTranspose1D(in=32, out=12, kernel=7, stride=2, padding=3, output_padding=1)
Input:  (64, 32, 500)
Output: (64, 12, ~1000-1002)  ← May be slightly longer
No activation (linear output)
```

**Length Correction:**
```python
# Transposed convolutions can produce longer outputs
decoded = decoded[:, :, :x.shape[2]]  # Trim to original length
decoded: (64, 12, 1000)
```

### 2.4 Output Format

```python
# Transpose back to (B, T, C) format for compatibility
output = decoded.transpose(1, 2)
output: (64, 1000, 12)
```

**Note:** CAE returns only reconstruction (no latent statistics like VAE).

---

## 3. Parameters and Hyperparameters

### 3.1 Model Parameters

```python
model = CAE(in_channels=12)
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| `in_channels` | 12 | Number of ECG leads (fixed) |

**No hyperparameters!** CAE has fixed architecture (simplicity is key).

### 3.2 Parameter Count Breakdown

| Component | Parameters | Calculation |
|-----------|-----------|-------------|
| **Encoder Conv1** | 10,144 | `(7*12+1)*32` = 2,689 → weights + bias |
| **Encoder Conv2** | 102,464 | `(5*32+1)*64` = 10,241 → weights + bias |
| **Encoder Conv3** | 245,888 | `(3*64+1)*128` = 24,641 → weights + bias |
| **Decoder ConvT1** | 245,824 | `(3*128+1)*64` = 24,641 → weights + bias |
| **Decoder ConvT2** | 102,432 | `(5*64+1)*32` = 10,241 → weights + bias |
| **Decoder ConvT3** | 10,764 | `(7*32+1)*12` = 2,689 → weights + bias |
| **Total** | **~250K** | Exact count depends on bias inclusion |

**Comparison:**
- CAE: 250K
- VAE-GRU: 500K (2x larger)
- BA-VAE: 600K (2.4x larger)
- VAE-BiLSTM-Attn: 2.1M (8.4x larger)
- ST-VAE: 8.2M (32.8x larger)

---

## 4. ECG Data Handling

### 4.1 Input Shape Transformation

**Training Pipeline Format:**
```
WindowDataset yields: (T, C) per sample
DataLoader batches:    (B, T, C) = (64, 1000, 12)
```

**CAE Requires:**
```
Conv1D expects: (B, C, T) = (64, 12, 1000)
```

**Handled by training.py:**
```python
x = x.transpose(1, 2)  # (B, T, C) → (B, C, T)
x_recon = model(x)
x = x.transpose(1, 2)  # Transpose back for loss computation
```

### 4.2 Preprocessing (Same as Other Models)

**PTB-XL Pipeline:**
1. **Bandpass filter:** 0.5-40 Hz
2. **Notch filter:** 50 Hz
3. **Z-score normalization:** Per lead
4. **Windowing:** 1000 samples (2 sec), stride 125

**No model-specific preprocessing needed.**

### 4.3 Normalization Impact

**Why z-score normalization matters:**
- Conv1D operates across time dimension
- Different leads have different amplitude scales (e.g., V2 >> aVR)
- Normalization ensures all leads contribute equally to learning

**Value Range:**
```
Normalized: ~[-3, +3] (99.7% of data within ±3σ)
```

---

## 5. Loss Function

### 5.1 Reconstruction Loss (MSE Only)

**Formula:**
```
L = MSE(x, x_recon)
  = (1/B) * Σ_i MSE(x_i, x_recon_i)
  = (1/B) * Σ_i Σ_c Σ_t (x[i,c,t] - x_recon[i,c,t])²
```

Where:
- `B` = Batch size (64)
- `C` = Channels/leads (12)
- `T` = Timesteps (1000)

**Code (from training.py):**
```python
loss_fn = torch.nn.MSELoss()
loss = loss_fn(x_recon, x)
```

**No KL divergence!** CAE is not a VAE.

### 5.2 Why MSE?

**Advantages:**
- ✅ Simple, interpretable
- ✅ Fast to compute
- ✅ Stable gradients
- ✅ Direct reconstruction penalty

**Limitations:**
- ❌ Assumes Gaussian noise (not always true for ECG)
- ❌ Equally weights all time points (clinically, some are more important)
- ❌ No probabilistic uncertainty estimate

### 5.3 Alternative Losses (Not Used)

**1. L1 (MAE):**
```python
loss = F.l1_loss(x_recon, x)
```
- More robust to outliers
- May improve on noisy ECGs

**2. Huber Loss:**
```python
loss = F.smooth_l1_loss(x_recon, x)
```
- Combines L1 and L2 benefits
- Less sensitive to large errors

**3. Perceptual Loss:**
```python
loss = MSE(VGG(x), VGG(x_recon))
```
- Uses pretrained features (e.g., VGG, ResNet)
- Captures semantic similarity
- Not commonly used for 1D signals

---

## 6. Anomaly Score Computation

### 6.1 Reconstruction Error (Same as Other Models)

**Per-Sample Score:**
```python
score(x) = MSE(x, x_recon)
         = (1 / C*T) * Σ_c Σ_t (x[c,t] - x_recon[c,t])²
```

**Code:**
```python
with torch.no_grad():
    x = x.transpose(1, 2)  # (B, T, C) → (B, C, T)
    x_recon = model(x)
    x = x.transpose(1, 2)  # Back to (B, T, C)
    x_recon = x_recon.transpose(1, 2)
    
    mse = torch.mean((x - x_recon) ** 2, dim=[1, 2])  # Per sample
    anomaly_scores = mse.cpu().numpy()
```

### 6.2 Threshold Selection (95th Percentile)

```python
# Compute on NORMAL ECGs
normal_errors = [score(x) for x in normal_dataloader]
threshold = np.percentile(normal_errors, 95)

# Classify
def predict(x):
    if score(x) > threshold:
        return "ABNORMAL"
    else:
        return "NORMAL"
```

### 6.3 Why Reconstruction Error Works

**Intuition:**
1. Model trained on **mostly normal** ECGs
2. Learns to reconstruct **normal patterns** well
3. **Abnormal patterns** are novel → high reconstruction error
4. Threshold separates normal (low error) from abnormal (high error)

**Limitation:**
- If abnormality is common in training data, model learns it
- Threshold-based detection is **unsupervised** (no labels used during training)

---

