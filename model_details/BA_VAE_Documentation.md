# Beat-Aligned VAE (BA-VAE) - Complete Technical Documentation

**Model Type:** Variational Autoencoder with Beat-Level Latent Representation  
**Architecture:** Bidirectional GRU Encoder-Decoder  
**Application:** ECG Anomaly Detection (Beat-Level Analysis)  
**Parameters:** ~600K (3.5M with expanded representations)

---

## 1. Architecture Overview

### 1.1 Design Philosophy

The Beat-Aligned VAE (BA-VAE) is designed around the concept of **per-beat encoding**, where each complete heartbeat window is compressed into a single latent representation. Unlike per-timestep VAEs (which create latent codes for every time sample), BA-VAE captures the holistic properties of a beat—making it ideal for:

- **Arrhythmia detection** (irregular beat patterns)
- **Beat classification** (normal vs abnormal morphology)
- **Efficient compression** (smaller latent space)
- **Clinical interpretability** (beat-level is natural for cardiologists)

### 1.2 Full Architecture Diagram

```
Input: (Batch, 1000 timesteps, 12 leads)
         ↓
   [ENCODER]
         ↓
   Bidirectional GRU (2 layers, 256 hidden)
   Input: 12 → Hidden: 256*2 (forward + backward)
         ↓
   LayerNorm(512)  ← Stabilization
         ↓
   Take Last Timestep h_last: (Batch, 512)
         ↓
   MLP Statistics Network:
     Linear(512 → 256)
     LayerNorm(256)
     GELU activation
     Dropout(0.1)
     Linear(256 → 64)  [mu, logvar concat]
         ↓
   Split → mu: (Batch, 32)
           logvar: (Batch, 32)
         ↓
   [REPARAMETERIZATION]
         ↓
   z = mu + eps * exp(0.5 * logvar)
   eps ~ N(0, 1)
   z: (Batch, 32) ← Per-beat latent code
         ↓
   [DECODER]
         ↓
   Latent to Initial Hidden:
     Linear(32 → 256)
     LayerNorm(256)
     Tanh activation
     h0: (1, Batch, 256)
         ↓
   Expand z to sequence:
     z_in: (Batch, 1000, 32)  ← Repeat latent across time
         ↓
   GRU Decoder (1 layer, 256 hidden)
   Input: 32, Hidden: 256
     d_seq: (Batch, 1000, 256)
         ↓
   LayerNorm(256)
         ↓
   Reconstruction MLP (per timestep):
     Flatten: (Batch*1000, 256)
     Linear(256 → 256)
     LayerNorm(256)
     GELU activation
     Dropout(0.1)
     Linear(256 → 24)  [mean + logvar for 12 leads]
     Split → x_mean: (Batch, 1000, 12)
             x_logvar: (Batch, 1000, 12)
         ↓
   Output: Reconstructed ECG (Batch, 1000, 12)
```

### 1.3 Key Architectural Features

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Encoder Type** | Bidirectional GRU (2 layers) | Captures both forward and backward temporal context |
| **Hidden Dimension** | 256 (512 bidirectional) | Large capacity for complex ECG patterns |
| **Latent Dimension** | 32 | Compact per-beat representation |
| **Decoder Type** | Unidirectional GRU (1 layer) | Generates sequence from beat-level latent |
| **Activation** | GELU | Smooth gradients, better optimization |
| **Normalization** | LayerNorm (encoder, decoder, MLPs) | Training stability |
| **Regularization** | Dropout (0.1), Free Bits (0.0 default) | Prevents overfitting and posterior collapse |

---

## 2. Forward Pass: Step-by-Step

### 2.1 Input Processing

```python
Input x: (Batch=64, Time=1000, Leads=12)
```

**Data Flow:**
1. ECG window contains ~2 seconds of 12-lead data (500 Hz sampling)
2. Pre-normalized (z-score) per lead
3. Shape: (B, T, C) where B=batch, T=timesteps, C=channels/leads

### 2.2 Encoding Phase

**Step 1: Bidirectional GRU Encoding**
```
h_seq, h_n = GRU_encoder(x)
h_seq: (64, 1000, 512)  ← 256*2 (forward + backward)
```

- **Forward pass** (timesteps 0→999): Captures past context
- **Backward pass** (timesteps 999→0): Captures future context
- **Concatenated**: Each timestep sees full temporal context

**Step 2: Layer Normalization**
```
h_seq = LayerNorm(h_seq)
```
- Normalizes across hidden dimension
- Stabilizes gradients during training

**Step 3: Extract Final Representation**
```
h_last = h_seq[:, -1, :]  # Take last timestep
h_last: (64, 512)
```
- Uses the final bidirectional hidden state
- Summarizes entire beat window into single vector

**Step 4: Compute Latent Statistics**
```
MLP: h_last → [mu, logvar]
  Linear(512 → 256) + LayerNorm + GELU + Dropout
  Linear(256 → 64)
  
stats: (64, 64)
mu, logvar = stats.chunk(2, dim=-1)
mu: (64, 32)
logvar: (64, 32)  ← Clamped to [-10, 10]
```

### 2.3 Reparameterization Trick

```
std = exp(0.5 * logvar)  # Convert log-variance to std dev
eps = randn_like(std)    # Sample from N(0, 1)
z = mu + eps * std       # Sample from N(mu, std^2)
z: (64, 32)              ← Latent code for each beat
```

**Why this trick?**
- Makes sampling operation **differentiable**
- Gradients flow through `mu` and `logvar`
- Enables end-to-end training

### 2.4 Decoding Phase

**Step 1: Initialize Decoder Hidden State**
```
h0 = Tanh(LayerNorm(Linear(z)))
h0: (1, 64, 256)  ← Initial hidden state from latent
```

**Step 2: Expand Latent to Sequence**
```
z_in = z.unsqueeze(1).expand(-1, 1000, -1)
z_in: (64, 1000, 32)  ← Repeat same latent at each timestep
```
- Decoder sees the same beat-level code at every step
- GRU generates temporal dynamics from this static input

**Step 3: GRU Decoding**
```
d_seq, _ = GRU_decoder(z_in, h0)
d_seq: (64, 1000, 256)
d_seq = LayerNorm(d_seq)
```

**Step 4: Reconstruct Per-Timestep**
```
d_flat = d_seq.reshape(64*1000, 256)
rec_stats = MLP(d_flat)  # Linear → LayerNorm → GELU → Dropout → Linear
rec_stats: (64*1000, 24)

x_mean, x_logvar = rec_stats.chunk(2, dim=-1)
x_mean = x_mean.reshape(64, 1000, 12)
x_logvar = x_logvar.reshape(64, 1000, 12)
```

**Output:**
- `x_mean`: Mean of reconstructed ECG
- `x_logvar`: Log-variance (uncertainty per timestep/lead)

### 2.5 Compatibility Expansion (for Training Pipeline)

```python
# Expand per-beat latent to per-timestep format
mu_expanded = mu.unsqueeze(1).expand(-1, 1000, -1)
logvar_expanded = logvar.unsqueeze(1).expand(-1, 1000, -1)

# Placeholder attention weights (uniform)
attn_weights = ones(64, 1000, 1000) / 1000
lead_w = ones(64, 1000, 12) / 12

return x_mean, x_logvar, mu_expanded, logvar_expanded, attn_weights, lead_w
```

**Why expand?**
- Training pipeline expects per-timestep latents for diagnostics
- Allows reuse of existing evaluation code
- Attention weights are placeholders (BA-VAE doesn't use attention)

---

## 3. Parameters and Hyperparameters

### 3.1 Model Parameters

```python
model = BeatVAE(
    n_leads=12,              # Number of ECG leads
    beat_len=1000,           # Timesteps per beat window
    enc_hidden=256,          # GRU hidden size (512 bidirectional)
    n_layers=2,              # GRU encoder layers
    latent_dim=32,           # Latent space dimension
    beta=0.3,                # KL divergence weight
    dropout=0.1              # Dropout rate
)
```

| Parameter | Value | Impact |
|-----------|-------|--------|
| `n_leads` | 12 | Fixed (standard 12-lead ECG) |
| `beat_len` | 1000 | ~2 seconds at 500 Hz |
| `enc_hidden` | 256 | Larger = more capacity, slower training |
| `n_layers` | 2 | Deeper = better feature extraction |
| `latent_dim` | 32 | Larger = more expressiveness, risk of collapse |
| `beta` | 0.3 | Higher = stronger regularization |
| `dropout` | 0.1 | Higher = more regularization, less overfitting |

### 3.2 Parameter Count Breakdown

| Component | Parameters | Calculation |
|-----------|-----------|-------------|
| **Encoder GRU** | ~1.05M | `3 * (12 + 256 + 1) * 256 * 2 layers * 2 directions` |
| **Encoder MLP** | ~262K | `512*256 + 256*64` |
| **Decoder Init** | 8.2K | `32*256` |
| **Decoder GRU** | ~329K | `3 * (32 + 256 + 1) * 256` |
| **Decoder MLP** | ~131K | `256*256 + 256*24` |
| **LayerNorms** | ~3K | Small bias/scale parameters |
| **Total** | **~1.78M** | Compact compared to attention models |

**Note:** Training reports ~3.5M due to expanded representations and optimizer states.

---

## 4. ECG Data Handling

### 4.1 Input Shape and Preprocessing

**Raw PTB-XL ECG:**
- Shape: (5000, 12) or (12, 5000)
- Sampling rate: 500 Hz
- Duration: 10 seconds

**Preprocessing Pipeline:**
1. **Bandpass filter:** 0.5-40 Hz (remove baseline wander and noise)
2. **Notch filter:** 50 Hz (remove powerline interference)
3. **Z-score normalization:** Per lead, mean=0, std=1
4. **Windowing:** Segment into 1000-sample windows (2 seconds)
   - Stride: 125 samples (0.25 seconds) → Overlapping windows
   - From 5000 samples → ~31 windows per ECG

**Model Input Format:**
```
Window shape: (Batch, 1000, 12)
Value range: ~[-3, +3] (z-score normalized)
No NaN values (handled by dataset)
```

### 4.2 Windowing Strategy

```python
from data_processing.window_dataset import WindowDataset

dataset = WindowDataset(
    root_dir='data/processed/ptbxl',
    window_size=1000,
    stride=125,
    n_leads=12
)
```

**Rationale:**
- 1000 samples (2 sec) captures 2-3 complete heartbeats
- Stride=125 creates overlapping windows for better coverage
- Each ECG generates multiple training samples

### 4.3 Normalization Details

**Per-Lead Z-Score:**
```python
x_lead = (x_lead - mean_lead) / (std_lead + eps)
```

- Applied **before** windowing
- Ensures each lead has comparable scale
- Prevents model bias toward high-amplitude leads (e.g., V2, V3)

**Benefits:**
- Stable gradients during training
- Faster convergence
- Consistent across different ECG datasets

### 4.4 Shape Transformations Through Model

```
Input:          (64, 1000, 12)   ← Batch of ECG windows
Encoder GRU:    (64, 1000, 512)  ← Bidirectional hidden states
Latent:         (64, 32)         ← Per-beat compression
Expanded:       (64, 1000, 32)   ← For decoder input
Decoder GRU:    (64, 1000, 256)  ← Reconstructed hidden states
Output:         (64, 1000, 12)   ← Reconstructed ECG
```

---

## 5. Loss Function

### 5.1 Total Loss Formulation

```
Total Loss = Reconstruction Loss + β * KL Divergence

L_total = L_recon + β * L_KL
```

### 5.2 Reconstruction Loss (MSE)

**Formula:**
```
L_recon = (1/B) * Σ_i MSE(x_i, x_mean_i)
        = (1/B) * Σ_i Σ_t Σ_l (x[i,t,l] - x_mean[i,t,l])²
```

Where:
- `B` = Batch size (64)
- `i` = Sample index
- `t` = Timestep (1000)
- `l` = Lead (12)

**Why MSE instead of Gaussian NLL?**
- **Stable training**: MSE doesn't depend on predicted variance
- **Prevents negative loss**: High x_logvar predictions can make NLL negative
- **Common practice**: Most VAE implementations use MSE for image/signal reconstruction

**Code:**
```python
recon_loss = F.mse_loss(x_mean, x, reduction='sum') / x.size(0)
```

### 5.3 KL Divergence

**Formula:**
```
L_KL = (1/B) * Σ_i Σ_d KL(q(z_d|x_i) || p(z_d))
     = (1/B) * Σ_i Σ_d [-0.5 * (1 + logvar[i,d] - mu[i,d]² - exp(logvar[i,d]))]
```

Where:
- `d` = Latent dimension (32)
- `q(z|x)` = Approximate posterior (encoder)
- `p(z)` = Prior N(0, 1)

**Closed-form KL for Gaussians:**
```
KL(N(mu, sigma²) || N(0, 1)) = 0.5 * (sigma² + mu² - 1 - log(sigma²))
                              = 0.5 * (exp(logvar) + mu² - 1 - logvar)
```

**Code:**
```python
# Use first timestep (all are the same after expansion)
mu = mu[:, 0, :]       # (B, 32)
logvar = logvar[:, 0, :]  # (B, 32)

# Clamp for numerical stability
logvar = torch.clamp(logvar, min=-10.0, max=10.0)

# KL per dimension
kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

# Optional: Free bits (prevent collapse)
if free_bits > 0:
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

kl_loss = kl_per_dim.sum(dim=-1).mean()  # Sum over dims, mean over batch
```

### 5.4 Beta (β) Annealing

**Cyclical Schedule:**
```python
def cyclical_annealing_beta(epoch, cycle_period=10, ramp_ratio=0.5, max_beta=0.3):
    cycle_epoch = (epoch - 1) % cycle_period
    ramp_epochs = max(1, int(cycle_period * ramp_ratio))
    
    if cycle_epoch < ramp_epochs:
        return max_beta * (cycle_epoch + 1) / ramp_epochs
    return max_beta
```

**Example (max_beta=0.3, cycle=10):**
```
Epoch:  1   2   3   4   5   6   7   8   9  10  11  12  13 ...
Beta: 0.06 0.12 0.18 0.24 0.30 0.30 0.30 0.30 0.30 0.30 0.06 0.12 0.18 ...
       └─── Ramp ───┘ └──────── Plateau ─────────┘ └─── Ramp ──→
```

**Why Cyclical?**
- **Prevents posterior collapse**: Low β allows model to learn reconstruction first
- **Encourages exploration**: Periodic resets help escape local minima
- **Empirically better**: Shown to improve VAE training across domains

### 5.5 Free Bits Regularization

**Purpose:** Ensure each latent dimension remains "active" (not collapsed to prior)

**Implementation:**
```python
kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
```

**Default:** `free_bits = 0.0` (disabled)

**When to use:**
- If training shows many "low-KL dimensions" (<0.001)
- If reconstructions look good but anomaly detection fails
- Typical value: 0.01 to 0.1

---

## 6. Anomaly Score Computation

### 6.1 Reconstruction Error

**Per-Sample Anomaly Score:**
```
score(x) = MSE(x, x_recon)
         = (1 / T*L) * Σ_t Σ_l (x[t,l] - x_mean[t,l])²
```

Where:
- `T` = 1000 timesteps
- `L` = 12 leads

**Code:**
```python
with torch.no_grad():
    x_mean, _, _, _, _, _ = model(x)
    mse = torch.mean((x - x_mean) ** 2, dim=[1, 2])  # Per sample
    anomaly_scores = mse.cpu().numpy()
```

### 6.2 Threshold Selection

**Method:** Percentile of normal ECG errors

```python
# Compute errors on NORMAL ECGs only
normal_errors = []
for x_normal in normal_dataloader:
    scores = compute_anomaly_score(model, x_normal)
    normal_errors.extend(scores)

# Set threshold (default: 95th percentile)
threshold = np.percentile(normal_errors, 95)
```

**Interpretation:**
- 95th percentile: 5% of normals flagged as false positives
- Higher percentile (99): More false positives, higher recall
- Lower percentile (90): Fewer false positives, lower recall

### 6.3 Binary Classification

```python
def predict(x):
    score = compute_anomaly_score(model, x)
    if score > threshold:
        return "ABNORMAL"
    else:
        return "NORMAL"
```

### 6.4 Alternative Scores (Not Currently Used)

**1. Latent Space Distance:**
```python
# Distance from latent mean to origin
latent_score = torch.norm(mu, dim=-1)
```

**2. KL Divergence:**
```python
# KL per sample
kl_score = kl_per_dim.sum(dim=-1)
```

**3. Combined Score:**
```python
# Weighted combination
combined = alpha * recon_score + (1-alpha) * latent_score
```

**Current Implementation:** Uses MSE reconstruction only (simplest, most effective)

---

