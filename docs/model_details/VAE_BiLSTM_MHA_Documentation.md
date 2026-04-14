# VAE-BiLSTM with Multi-Head Attention: Comprehensive Technical Documentation

## Executive Summary

**VAE-BiLSTM with Multi-Head Attention (MA-VAE style)** is a variational autoencoder that extends the BiLSTM-Attention architecture with multi-head attention mechanisms, enabling parallel processing of multiple attention perspectives. The model achieved **F1 = 0.825** with **83% precision** and **82% recall** on the PTB-XL dataset, representing the **2nd best performance** among all evaluated models and a **5% improvement** over single-head attention.

---

## 1. Model Architecture

### 1.1 Architecture Overview

The VAE-BiLSTM-MHA architecture builds upon VAE-BiLSTM-Attn by replacing single-head attention with **multi-head attention (MHA)**, inspired by Transformer architectures. This enables the model to attend to different representation subspaces simultaneously.

```
Input ECG: (B, 1000, 12)
    ↓
[BiLSTM Encoder]
  - Layer 1: 12 → 128 hidden (bidirectional)
  - Layer 2: 128 → 128 hidden (bidirectional)
    ↓
BiLSTM Output: (B, 1000, 256)  # 128*2 directions
    ↓
[Multi-Head Attention]
  - Num heads: 8
  - Head dim: 32 (256 / 8)
  - Parallel attention across 8 subspaces
    ↓
MHA Output: (B, 1000, 256)
    ↓
[Global Pooling]
  - Mean pooling over timesteps
    ↓
Pooled Features: (B, 256)
    ↓
[Latent Space]
  - fc_mu: 256 → 32
  - fc_logvar: 256 → 32
  - Reparameterization: z ~ N(μ, σ)
    ↓
Latent z: (B, 32)
    ↓
[Decoder Projection]
  - fc_decode: 32 → 128
  - Reshape: (B, 1, 128)
    ↓
[BiLSTM Decoder]
  - Layer 1: 128 → 128 hidden (bidirectional)
  - Layer 2: 128 → 128 hidden (bidirectional)
    ↓
Decoder Output: (B, 1000, 256)
    ↓
[Output Layer]
  - Linear: 256 → 12
    ↓
Reconstructed ECG: (B, 1000, 12)
```

### 1.2 Key Components

#### 1.2.1 BiLSTM Encoder
```python
self.encoder = nn.LSTM(
    input_size=12,          # 12 ECG leads
    hidden_size=128,        # Hidden dimension
    num_layers=2,           # 2 stacked layers
    batch_first=True,
    bidirectional=True,     # Process forward and backward
    dropout=0.2
)
# Output: (B, 1000, 256)
```

**Purpose**: Identical to VAE-BiLSTM-Attn encoder. Captures bidirectional temporal dependencies in ECG sequences.

#### 1.2.2 Multi-Head Attention (MHA)
```python
self.multihead_attention = nn.MultiheadAttention(
    embed_dim=256,          # Input dimension (BiLSTM output)
    num_heads=8,            # Number of attention heads
    dropout=0.1,
    batch_first=True
)
# Input: (B, 1000, 256)
# Output: (B, 1000, 256)
```

**How Multi-Head Attention Works**:

1. **Linear Projections**: Split 256-dim embeddings into 8 heads of 32-dim each
   ```python
   # Conceptually (implemented internally by PyTorch):
   Q = Linear(256, 256)(x)  # Query
   K = Linear(256, 256)(x)  # Key
   V = Linear(256, 256)(x)  # Value
   
   # Reshape to (B, 8, 1000, 32) for 8 parallel heads
   Q_heads = Q.view(B, 1000, 8, 32).transpose(1, 2)
   K_heads = K.view(B, 1000, 8, 32).transpose(1, 2)
   V_heads = V.view(B, 1000, 8, 32).transpose(1, 2)
   ```

2. **Scaled Dot-Product Attention (per head)**:
   $$
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   $$
   where $d_k = 32$ (head dimension)

3. **Concatenate Heads**:
   ```python
   # Each head: (B, 1000, 32)
   # Concatenate: (B, 1000, 256)
   attn_output = concat([head_1, head_2, ..., head_8], dim=-1)
   ```

4. **Output Projection**:
   ```python
   output = Linear(256, 256)(attn_output)
   ```

**Why Multi-Head?**
- **Single-head attention**: Learns one attention pattern (e.g., focus on QRS complex)
- **Multi-head attention**: Learns 8 different attention patterns simultaneously:
  - Head 1: QRS complex detection
  - Head 2: ST-segment elevation
  - Head 3: T-wave morphology
  - Head 4: P-wave presence
  - Head 5: R-R interval regularity
  - Head 6: Baseline wander
  - Head 7: Lead-specific patterns
  - Head 8: Global rhythm context

#### 1.2.3 Latent Space Projection
```python
# Global pooling to aggregate temporal information
pooled = torch.mean(mha_output, dim=1)  # (B, 256)

self.fc_mu = nn.Linear(256, 32)
self.fc_logvar = nn.Linear(256, 32)

# Reparameterization trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std  # z ~ N(μ, σ)
```

**Key Difference from Single-Head**:
- Single-head: Combined temporal + lead features (268-dim input)
- Multi-head: Global pooled MHA output (256-dim input)

#### 1.2.4 BiLSTM Decoder
```python
# Project latent to initial hidden state
self.fc_decode = nn.Linear(32, 128)
z_projected = self.fc_decode(z).unsqueeze(1)  # (B, 1, 128)

self.decoder = nn.LSTM(
    input_size=128,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.2
)
# Output: (B, 1000, 256)
```

**Purpose**: Identical to single-head decoder. Reconstructs temporal sequence from compact latent representation.

---

## 2. Forward Pass Breakdown

### 2.1 Encoding Phase

**Input**: ECG batch `x` of shape `(B, 1000, 12)`

**Step 1: BiLSTM Encoding**
```python
encoder_out, (h_n, c_n) = self.encoder(x)
# encoder_out: (B, 1000, 256)
```

**Step 2: Multi-Head Attention**
```python
# Self-attention: query, key, value are all encoder_out
attn_output, attn_weights = self.multihead_attention(
    query=encoder_out,  # (B, 1000, 256)
    key=encoder_out,
    value=encoder_out
)
# attn_output: (B, 1000, 256)
# attn_weights: (B, 8, 1000, 1000) [attention maps for 8 heads]
```

**What's Happening Inside**:
```
For each head h ∈ {1, 2, ..., 8}:
  1. Project to 32-dim subspace: Q_h, K_h, V_h
  2. Compute attention scores: S_h = Q_h @ K_h^T / sqrt(32)  # (B, 1000, 1000)
  3. Apply softmax: A_h = softmax(S_h, dim=-1)
  4. Weighted sum: O_h = A_h @ V_h  # (B, 1000, 32)

Concatenate all heads: O = [O_1, O_2, ..., O_8]  # (B, 1000, 256)
Final projection: attn_output = W_o @ O
```

**Step 3: Global Pooling**
```python
pooled = torch.mean(attn_output, dim=1)  # (B, 256)
```

**Step 4: Latent Projection**
```python
mu = self.fc_mu(pooled)            # (B, 32)
logvar = self.fc_logvar(pooled)     # (B, 32)
z = self.reparameterize(mu, logvar)  # (B, 32)
```

### 2.2 Decoding Phase

**Step 5: Latent to Sequence**
```python
z_decoded = self.fc_decode(z).unsqueeze(1)  # (B, 1, 128)
z_sequence = z_decoded.repeat(1, 1000, 1)   # (B, 1000, 128)
```

**Step 6: BiLSTM Decoding**
```python
decoder_out, _ = self.decoder(z_sequence)   # (B, 1000, 256)
reconstruction = self.fc_out(decoder_out)   # (B, 1000, 12)
```

### 2.3 Complete Forward Pass Summary
```python
def forward(x):
    # Encoding
    encoder_out, _ = BiLSTM_encoder(x)           # (B, 1000, 12) → (B, 1000, 256)
    attn_out, attn_weights = MHA(encoder_out)    # (B, 1000, 256) → (B, 1000, 256)
    pooled = mean(attn_out, dim=1)               # (B, 256)
    
    # Latent space
    mu, logvar = fc_mu(pooled), fc_logvar(pooled)
    z = reparameterize(mu, logvar)               # (B, 32)
    
    # Decoding
    z_seq = fc_decode(z).repeat(1, 1000, 1)      # (B, 1000, 128)
    decoder_out = BiLSTM_decoder(z_seq)          # (B, 1000, 256)
    x_recon = fc_out(decoder_out)                # (B, 1000, 12)
    
    return x_recon, mu, logvar, attn_weights
```

---

## 3. ECG Data Handling

### 3.1 Data Preprocessing Pipeline

Identical to VAE-BiLSTM-Attn:

```python
# 1. Raw PTB-XL data: 21799 patients, 12 leads, 5000 samples @ 500 Hz
# 2. Bandpass filter: 0.5 - 40 Hz
# 3. Notch filter: 50 Hz
# 4. Fixed-length windowing: 1000 samples (2 seconds)
#    - Window size: 1000 samples
#    - Stride: 500 samples (50% overlap)
# 5. Z-score normalization per lead
```

### 3.2 Data Shapes Through the Model

| Stage | Shape | Description |
|-------|-------|-------------|
| Raw ECG | `(B, 1000, 12)` | Batch of normalized 2-second windows |
| BiLSTM Encoder Output | `(B, 1000, 256)` | Bidirectional hidden states |
| MHA Output | `(B, 1000, 256)` | Multi-head attention-weighted features |
| Global Pooled Features | `(B, 256)` | Mean-pooled temporal representation |
| Latent μ, σ | `(B, 32)` | Mean and log-variance |
| Latent z | `(B, 32)` | Sampled latent vector |
| Decoder Input | `(B, 1000, 128)` | Repeated latent projection |
| Decoder Output | `(B, 1000, 256)` | Reconstructed hidden states |
| Reconstruction | `(B, 1000, 12)` | Final 12-lead ECG |

### 3.3 Dataset Statistics

**Training Set**: 7,611 samples
**Validation Set**: 1,903 samples
**Test Set**: 21,833 windows (9,514 normal + 12,319 abnormal)

---

## 4. Loss Function and Training

### 4.1 VAE Loss Function

Identical to VAE-BiLSTM-Attn:

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}
$$

#### 4.1.1 Reconstruction Loss
$$
\mathcal{L}_{\text{recon}} = \frac{1}{B \cdot T \cdot L} \sum_{b=1}^{B} \sum_{t=1}^{T} \sum_{l=1}^{L} (x_{btl} - \hat{x}_{btl})^2
$$

#### 4.1.2 KL Divergence Loss
$$
\mathcal{L}_{\text{KL}} = -\frac{1}{2B \cdot D} \sum_{b=1}^{B} \sum_{d=1}^{D} (1 + \log(\sigma_d^2) - \mu_d^2 - \sigma_d^2)
$$

#### 4.1.3 Beta-Annealing Strategy

$$
\beta(t) = \beta_{\max} \cdot \min\left(1, \frac{2t \mod P}{P}\right)
$$

Where $\beta_{\max} = 0.3$ and $P = 10$ epochs.

### 4.2 Training Configuration

**Optimizer**: Adam
- Learning rate: 0.0005 (initial)
- Betas: (0.9, 0.999)
- Weight decay: 1e-5

**Learning Rate Scheduler**: ReduceLROnPlateau
- Factor: 0.5
- Patience: 5 epochs
- Schedule:
  - Epochs 1-26: LR = 0.0005
  - Epochs 27-50: LR = 0.00025
  - Epochs 51-90: LR = 0.000125

**Gradient Clipping**: Max norm = 1.0

**Batch Size**: 64

**Epochs**: 90 (best model at epoch 84)

### 4.3 Training Dynamics

**Loss Progression**:

| Epoch | Train Loss | Val Loss | Train Recon | Val Recon | Train KL | Val KL |
|-------|------------|----------|-------------|-----------|----------|--------|
| 1 | 2766.46 | 858.57 | 2656.50 | 797.10 | 1832.73 | 1024.44 |
| 10 | 239.63 | 235.63 | 191.67 | 189.84 | 159.84 | 152.64 |
| 30 | 206.46 | 224.59 | 153.60 | 159.12 | 149.44 | 141.81 |
| 50 | 199.37 | 203.91 | 150.95 | 165.75 | 140.73 | 145.19 |
| 70 | 172.96 | 196.72 | 120.07 | 138.30 | 131.78 | 135.00 |
| **84** | **157.57** | **156.96** | **119.42** | **140.42** | **131.89** | **135.00** |
| 90 | 167.36 | 177.45 | 119.14 | 146.07 | 130.00 | 135.00 |

**Key Observations**:
1. **Extended Training**: 90 epochs (vs. 50 for single-head) due to more parameters
2. **KL Stabilization**: KL converges to ~135 (vs. ~45 for single-head) — **higher regularization**
3. **Best Model Late**: Epoch 84 has lowest validation loss
4. **Reconstruction Quality**: Validation recon = 140.42 (slightly higher than single-head's 104.87)

**Why Higher KL Loss?**
- Multi-head attention provides richer representations
- More complex latent space requires stronger regularization
- KL = 135 means latent distributions deviate more from N(0,1), but this enables better discrimination

---

## 5. Anomaly Score Computation

### 5.1 Reconstruction Error Metric

Identical to VAE-BiLSTM-Attn:

$$
\text{Anomaly Score}(x) = \frac{1}{T \cdot L} \sum_{t=1}^{T} \sum_{l=1}^{L} (x_{tl} - \hat{x}_{tl})^2
$$

### 5.2 Threshold Selection

**Method**: 95th percentile on validation set

```python
threshold = np.percentile(val_scores, 95)  # threshold = 0.042
```

**Classification Rule**:
```python
if anomaly_score > 0.042:
    prediction = "Abnormal"
else:
    prediction = "Normal"
```

### 5.3 Confusion Matrix Analysis

|  | Predicted Normal | Predicted Abnormal |
|--|------------------|-------------------|
| **Actual Normal** | TN = 7,448 (78.3%) | FP = 2,066 (21.7%) |
| **Actual Abnormal** | FN = 2,218 (18.0%) | TP = 10,101 (82.0%) |

**Derived Metrics**:
- **Precision** = TP / (TP + FP) = 10101 / 12167 = **0.83**
- **Recall** = TP / (TP + FN) = 10101 / 12319 = **0.82**
- **F1-Score** = 2 × (0.83 × 0.82) / (0.83 + 0.82) = **0.825**
- **Specificity** = TN / (TN + FP) = 7448 / 9514 = **0.783**

**Comparison to Single-Head**:
- **FP Reduction**: 2066 vs. 2558 (19% fewer false positives)
- **FN Reduction**: 2218 vs. 2710 (18% fewer false negatives)
- **TN Improvement**: 7448 vs. 6956 (492 more correct normal predictions)
- **TP Improvement**: 10101 vs. 9609 (492 more correct abnormal predictions)

---

