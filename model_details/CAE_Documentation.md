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

## 7. Performance Analysis

### 7.1 Training Characteristics

**Typical Training Curve:**
```
Epoch 1-10:   Rapid MSE decrease (learning basic patterns)
Epoch 10-30:  Slow improvement (refinement)
Epoch 30-50:  Plateau (convergence)
```

**Target MSE:**
- Normal ECGs: 0.01 - 0.05
- Lower is better (but watch for overfitting)

**Note:** CAE typically requires **fewer epochs** than VAE (no KL annealing).

### 7.2 Inference Speed

| Model | Inference Time (ms/sample) | Speedup vs Baseline |
|-------|----------------------------|---------------------|
| **CAE** | **~0.5-1.0** | Fastest (baseline) |
| BA-VAE | 1.53 | 1.5-3x slower |
| HL-VAE | 3.72 | 3.7-7.4x slower |
| VAE-BiLSTM-Attn | 4.54 | 4.5-9x slower |

**Why Fastest?**
- ✅ No recurrent layers (GRU/LSTM) → Parallelizable
- ✅ No attention mechanism → Fewer operations
- ✅ No sampling/reparameterization → Deterministic
- ✅ Smaller model (250K params) → Less computation

### 7.3 Expected Performance (No Evaluation Yet)

**Prediction (based on literature):**
- **F1-Score:** 0.30 - 0.45 (lowest among models)
- **AUC-ROC:** 0.65 - 0.75 (moderate discrimination)
- **Precision:** 0.70 - 0.85 (depends on threshold)
- **Recall:** 0.20 - 0.40 (likely misses subtle abnormalities)

**Why Lower Performance?**
- ❌ No probabilistic modeling (VAE's latent space helps)
- ❌ No attention (can't focus on important features)
- ❌ No sequence modeling (GRU/LSTM capture temporal dependencies)
- ✅ But: Fast, simple, good baseline

### 7.4 Comparison Summary

| Model | Architecture | F1 (Expected) | Speed | Complexity |
|-------|--------------|---------------|-------|------------|
| **CAE** | Conv1D | **0.30-0.45** | **Fastest** | **Simplest** |
| BA-VAE | BiGRU + VAE | 0.55 | Fast | Low |
| VAE-BiLSTM-Attn | BiLSTM + Attn + VAE | 0.52 | Slow | High |
| ST-VAE | Dual CNN + FFT + VAE | 0.66 | Very Fast | High |

---

## 8. Why CAE is Useful

### 8.1 As a Baseline

**Essential for Research:**
1. **Comparison point:** How much do VAE, attention, RNN improve over simple CNN?
2. **Ablation study:** Does probabilistic modeling (VAE) matter?
3. **Complexity justification:** If CAE performs well, why use complex models?

**Example Thesis Claim:**
> "Our proposed VAE-BiLSTM-Attn model achieves 0.52 F1-score, a 30% improvement over the CAE baseline (0.40), demonstrating the value of attention mechanisms and variational inference."

### 8.2 For Deployment

**When to Use CAE:**
- ✅ **Strict latency requirements:** <1ms inference needed
- ✅ **Resource-constrained devices:** Edge devices, wearables
- ✅ **Simple abnormalities:** Gross morphological changes
- ✅ **High throughput:** Process 1000+ ECGs/sec

**When NOT to Use:**
- ❌ High-stakes detection (missed abnormalities costly)
- ❌ Subtle abnormalities (small ST changes, T-wave inversions)
- ❌ Need for uncertainty estimates (CAE is deterministic)

### 8.3 Interpretability

**What CAE Learns (Visualizable):**

1. **Encoder Filter Visualization:**
```python
# Layer 1 filters (12 → 32)
filters = model.encoder[0].weight.data  # (32, 12, 7)
# Plot: Each filter shows learned 7-sample pattern per lead
```

2. **Activation Maps:**
```python
# Intermediate activations
x_enc1 = F.relu(model.encoder[0](x))  # (B, 32, 500)
# Plot: Which time segments activate which filters?
```

3. **Reconstruction Comparison:**
```python
# Plot original vs reconstructed
plt.plot(x[0, :, 1], label='Original Lead II')
plt.plot(x_recon[0, :, 1], label='Reconstructed Lead II')
# See: Which features preserved? Which distorted?
```

**Limitation:** Less interpretable than attention weights (no explicit importance scores).

---

## 9. Receptive Field Analysis

### 9.1 Effective Receptive Field

**Layer-by-layer calculation:**

```
Input: 1000 samples

Layer 1: Conv1D(kernel=7, stride=2)
  Receptive field: 7 samples
  Output: 500 samples
  
Layer 2: Conv1D(kernel=5, stride=2)
  Receptive field: 7 + 2*(5-1) = 15 samples
  Output: 250 samples
  
Layer 3: Conv1D(kernel=3, stride=2)
  Receptive field: 15 + 4*(3-1) = 23 samples
  Output: 125 samples (bottleneck)
```

**Bottleneck Receptive Field:** 23 samples = **46 milliseconds** at 500 Hz

**What this means:**
- Each bottleneck feature "sees" ~46ms of ECG
- Shorter than typical QRS duration (~80-120ms)
- May miss long-range dependencies (e.g., QRS-T relationship)

### 9.2 Comparison with RNN Models

| Model | Effective Receptive Field | Coverage |
|-------|---------------------------|----------|
| **CAE** | 23 samples (46 ms) | **Local features only** |
| GRU/LSTM | Entire sequence (2000 ms) | **Full temporal context** |
| Attention | Any timestep pair | **Global context** |

**Implication:** CAE may struggle with abnormalities requiring global context (e.g., long QT syndrome).

---

## 10. Training Dynamics

### 10.1 Loss Monitoring

**Healthy Training:**
```
Epoch 1:   MSE = 0.50 (random initialization)
Epoch 10:  MSE = 0.10 (rapid learning)
Epoch 30:  MSE = 0.03 (refinement)
Epoch 50:  MSE = 0.02 (convergence)
```

**Red Flags:**
- **MSE > 0.1 after 30 epochs:** Underfitting, increase model capacity
- **Train MSE << Val MSE:** Overfitting, add regularization (dropout, weight decay)
- **MSE oscillates wildly:** Learning rate too high

### 10.2 Optimizer Settings

**Recommended (from training.py):**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```

**Typical Schedule:**
```
Epoch 1-30:  lr = 5e-4 (default)
Epoch 31-40: lr = 2.5e-4 (if plateau)
Epoch 41-50: lr = 1.25e-4 (if plateau again)
```

### 10.3 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Why needed:**
- Transposed convolutions can have exploding gradients
- Clipping ensures stable training

---

## 11. Limitations and Failure Modes

### 11.1 Known Limitations

**1. No Uncertainty:**
- Outputs single reconstruction (deterministic)
- Can't express "I'm uncertain about this reconstruction"
- VAEs provide uncertainty via sampling

**2. Limited Receptive Field:**
- Only sees 46ms context at bottleneck
- Misses long-range dependencies (QRS ↔ T-wave)
- RNNs/Transformers handle this better

**3. No Attention:**
- Treats all time points equally
- Can't focus on clinically important segments (QRS, ST, T-wave)
- Attention models explicitly learn importance

**4. Shallow Architecture:**
- Only 3 encoder + 3 decoder layers
- May lack capacity for complex patterns
- Deeper models (ResNet, U-Net) perform better

### 11.2 When CAE Fails

**Scenario 1: Subtle ST-Elevation MI**
```
Normal: Flat ST segment
Abnormal: 0.5mm elevation (small change)
CAE: Reconstructs as flat (within noise tolerance)
Result: Missed detection (False Negative)
```

**Scenario 2: Intermittent Arrhythmia**
```
Normal: Regular rhythm (R-R interval = 800ms)
Abnormal: Occasional PVC (R-R = 600ms)
CAE: Trained on mostly normal beats, reconstructs as normal
Result: Missed detection (False Negative)
```

**Scenario 3: Baseline Wander**
```
Normal: Clean ECG
Input: ECG with residual baseline wander (artifact)
CAE: May learn to reconstruct artifact as "normal"
Result: False Negative or False Positive depending on artifact
```

---

## 12. Improvements and Variants

### 12.1 Architectural Improvements

**1. Deeper Network (CAE-Deep):**
```python
# Add more layers
self.encoder = nn.Sequential(
    nn.Conv1d(12, 32, 7, 2, 3), nn.ReLU(),
    nn.Conv1d(32, 64, 5, 2, 2), nn.ReLU(),
    nn.Conv1d(64, 128, 3, 2, 1), nn.ReLU(),
    nn.Conv1d(128, 256, 3, 2, 1), nn.ReLU(),  # Extra layer
)
```
- **Benefit:** Larger receptive field, more capacity
- **Cost:** Slower inference, more parameters

**2. Residual Connections (CAE-ResNet):**
```python
class ResBlock(nn.Module):
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity  # Skip connection
        return F.relu(out)
```
- **Benefit:** Better gradient flow, deeper networks
- **Used in:** ST-VAE model

**3. Attention Mechanism (CAE-Attn):**
```python
# Add channel or spatial attention
self.channel_attn = nn.Sequential(
    nn.AdaptiveAvgPool1d(1),
    nn.Conv1d(128, 128, 1),
    nn.Sigmoid()
)
```
- **Benefit:** Focus on important features
- **Cost:** More complexity, slower

**4. Dilated Convolutions:**
```python
nn.Conv1d(32, 64, kernel_size=5, dilation=2)
```
- **Benefit:** Larger receptive field without more parameters
- **Used in:** WaveNet, TCN models

### 12.2 Loss Improvements

**1. Weighted MSE:**
```python
# Weight clinically important regions (QRS, ST, T-wave)
weights = detect_clinical_segments(x)  # e.g., QRS gets 2x weight
loss = (weights * (x - x_recon)**2).mean()
```

**2. Multi-Scale Loss:**
```python
# Loss at multiple resolutions
loss = MSE(x, x_recon) + 0.5*MSE(downsample(x), downsample(x_recon))
```

**3. Adversarial Loss (CAE-GAN):**
```python
# Add discriminator
D_real = discriminator(x)
D_fake = discriminator(x_recon)
loss_G = MSE(x, x_recon) + lambda * BCE(D_fake, ones)
```
- **Benefit:** More realistic reconstructions
- **Cost:** Unstable training (GANs are tricky)

---

## 13. Use Cases and Recommendations

### 13.1 When to Use CAE

✅ **Baseline Comparison:**
- Research papers need simple baseline
- Demonstrate value of complex models

✅ **Fast Prototyping:**
- Quick experiments
- Iterate rapidly without long training

✅ **Extreme Speed Requirements:**
- Real-time monitoring (<1ms latency)
- High-throughput batch processing

✅ **Limited Compute:**
- Edge devices (smartphone, wearable)
- No GPU available

✅ **Interpretable Filters:**
- Visualize learned patterns
- Educational purposes

### 13.2 When NOT to Use CAE

❌ **High-Stakes Clinical Decision:**
- False negatives costly (missed MI, arrhythmia)
- Use ensemble or higher-performing model (ST-VAE)

❌ **Subtle Abnormalities:**
- Small ST changes, T-wave inversions
- Need attention or probabilistic models

❌ **Uncertainty Quantification:**
- Need confidence scores
- Use VAE models (sample latent space)

❌ **Long-Range Dependencies:**
- QT interval prolongation, rhythm abnormalities
- Use RNN/Transformer models

---

## 14. Reproducibility

### 14.1 Training Command

```powershell
python src/training.py `
  --model cae `
  --epochs 50 `
  --batch_size 64 `
  --lr 5e-4 `
  --dataset ptbxl
```

### 14.2 Evaluation Command

```powershell
python src/evaluate.py `
  --model cae `
  --batch-size 64 `
  --threshold-percentile 95
```

### 14.3 Expected Training Time

| Hardware | Batch Size | Time per Epoch | Total (50 epochs) |
|----------|-----------|----------------|-------------------|
| RTX 3060 | 64 | ~30 sec | **~25 min** |
| CPU (16 cores) | 64 | ~3 min | **~2.5 hours** |

**Fastest model to train!**

---

## 15. Summary

**Convolutional Autoencoder (CAE)** is the **simplest baseline** for ECG anomaly detection:

✅ **Fastest inference:** <1ms per sample  
✅ **Smallest model:** 250K parameters  
✅ **Simplest training:** MSE loss only, no KL annealing  
✅ **Good baseline:** Demonstrates value of complex models  
✅ **Parallelizable:** No recurrent dependencies  

**Limitations:**
❌ **Lowest performance:** Expected F1 ~0.30-0.45  
❌ **No uncertainty:** Deterministic reconstruction  
❌ **Limited receptive field:** 46ms context  
❌ **No attention:** Equal weight to all features  

**Recommendation:**
- Use for **baseline comparison** in research
- Use for **extreme speed** requirements
- **Not recommended** for clinical deployment (too many missed abnormalities)

---

*Document Version: 1.0*  
*Last Updated: December 1, 2025*  
*Model Version: CAE (Standard Architecture)*
