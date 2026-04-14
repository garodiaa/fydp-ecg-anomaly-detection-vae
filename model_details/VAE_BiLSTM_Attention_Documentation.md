# VAE-BiLSTM with Attention: Comprehensive Technical Documentation

## Executive Summary

**VAE-BiLSTM with Attention** is a variational autoencoder architecture that combines bidirectional LSTM encoders with dual attention mechanisms (temporal attention and lead-wise attention) to model 12-lead ECG data for anomaly detection. The model achieved **F1 = 0.785** with **79% precision** and **78% recall** on the PTB-XL dataset, demonstrating strong performance through its ability to capture both temporal dependencies and inter-lead relationships in cardiac signals.



---

## 1. Model Architecture

### 1.1 Architecture Overview

The VAE-BiLSTM-Attn architecture consists of:

1. **BiLSTM Encoder** (2 layers, bidirectional): Processes temporal sequences
2. **Temporal Attention**: Weights timesteps based on their relevance
3. **Lead-wise Attention**: Weights ECG leads based on their contribution
4. **Latent Space Projection**: Maps to 32-dimensional latent space (μ and σ)
5. **BiLSTM Decoder** (2 layers, bidirectional): Reconstructs ECG signals
6. **Output Projection**: Generates 12-lead ECG reconstruction

```
Input ECG: (B, 1000, 12)
    ↓
[BiLSTM Encoder]
  - Layer 1: 12 → 128 hidden (bidirectional)
  - Layer 2: 128 → 128 hidden (bidirectional)
    ↓
BiLSTM Output: (B, 1000, 256)  # 128*2 directions
    ↓
[Temporal Attention]
  - Attention weights: (B, 1000, 1)
  - Context vector: (B, 256)
    ↓
[Lead-wise Attention]
  - Applied to original input
  - Weighted leads: (B, 12)
    ↓
[Combined Features]
  - Concat(temporal_context, lead_features)
  - Combined: (B, 268)  # 256 + 12
    ↓
[Latent Space]
  - fc_mu: 268 → 32
  - fc_logvar: 268 → 32
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
# Output: (B, 1000, 256)  # 128 * 2 directions
```

**Purpose**: Captures bidirectional temporal dependencies in ECG sequences. Bidirectionality allows the model to see both past and future context for each timestep, crucial for detecting cardiac events that span multiple beats.

#### 1.2.2 Temporal Attention Mechanism
```python
self.temporal_attention = nn.Sequential(
    nn.Linear(256, 128),
    nn.Tanh(),
    nn.Linear(128, 1)
)
# Attention weights: (B, 1000, 1)
# Softmax over timesteps: attention_weights = softmax(scores, dim=1)
# Context: (B, 256) = sum(encoder_out * attention_weights)
```

**Purpose**: Learns to focus on diagnostically important timesteps (e.g., QRS complexes, ST segments, T-waves). Not all timesteps contribute equally to anomaly detection—R-peaks and ST segments are more informative than baseline intervals.

#### 1.2.3 Lead-wise Attention Mechanism
```python
self.lead_attention = nn.Sequential(
    nn.Linear(1000, 128),   # Process temporal dimension of each lead
    nn.Tanh(),
    nn.Linear(128, 1)
)
# Attention weights: (B, 12, 1)
# Softmax over leads: lead_weights = softmax(lead_scores, dim=1)
# Weighted leads: (B, 12) = sum(input * lead_weights)
```

**Purpose**: Learns to weight different ECG leads based on their diagnostic value. For example, leads V2-V4 (precordial) often show ST-elevation in MI, while lead II is optimal for rhythm analysis. The model learns to prioritize relevant leads.

#### 1.2.4 Latent Space Projection
```python
# Combined features: (B, 268) = concat(temporal_context, lead_features)
self.fc_mu = nn.Linear(268, 32)
self.fc_logvar = nn.Linear(268, 32)

# Reparameterization trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std  # z ~ N(μ, σ)
```

**Purpose**: Projects 268-dimensional attention-weighted features to 32-dimensional latent space. The reparameterization enables gradient flow during training while maintaining probabilistic sampling.

#### 1.2.5 BiLSTM Decoder
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

**Purpose**: Reconstructs temporal sequence from compact latent representation. Bidirectional processing ensures smooth, physiologically plausible ECG waveforms.

---

## 2. Forward Pass Breakdown

### 2.1 Encoding Phase

**Input**: ECG batch `x` of shape `(B, 1000, 12)`

**Step 1: BiLSTM Encoding**
```python
encoder_out, (h_n, c_n) = self.encoder(x)
# encoder_out: (B, 1000, 256)  # BiLSTM outputs
```

**Step 2: Temporal Attention**
```python
attention_scores = self.temporal_attention(encoder_out)  # (B, 1000, 1)
attention_weights = F.softmax(attention_scores, dim=1)   # Normalize over time
context_vector = torch.sum(encoder_out * attention_weights, dim=1)  # (B, 256)
```
- **Interpretation**: `attention_weights[b, t, 0]` represents the importance of timestep `t` in sample `b`. High values indicate diagnostically relevant regions (QRS, ST-segment).

**Step 3: Lead-wise Attention**
```python
x_transposed = x.transpose(1, 2)  # (B, 12, 1000)
lead_scores = self.lead_attention(x_transposed)  # (B, 12, 1)
lead_weights = F.softmax(lead_scores, dim=1)     # Normalize over leads
weighted_leads = torch.sum(x_transposed * lead_weights, dim=2)  # (B, 12)
```
- **Interpretation**: `lead_weights[b, l, 0]` shows the contribution of lead `l` in sample `b`. Example: V2-V4 may have higher weights for ischemia detection.

**Step 4: Feature Combination & Latent Projection**
```python
combined = torch.cat([context_vector, weighted_leads], dim=1)  # (B, 268)
mu = self.fc_mu(combined)            # (B, 32)
logvar = self.fc_logvar(combined)     # (B, 32)
z = self.reparameterize(mu, logvar)   # (B, 32)
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
    context = temporal_attention(encoder_out)    # (B, 256)
    lead_features = lead_attention(x)             # (B, 12)
    combined = concat([context, lead_features])   # (B, 268)
    
    # Latent space
    mu, logvar = fc_mu(combined), fc_logvar(combined)
    z = reparameterize(mu, logvar)               # (B, 32)
    
    # Decoding
    z_seq = fc_decode(z).repeat(1, 1000, 1)      # (B, 1000, 128)
    decoder_out = BiLSTM_decoder(z_seq)          # (B, 1000, 256)
    x_recon = fc_out(decoder_out)                # (B, 1000, 12)
    
    return x_recon, mu, logvar, attention_weights, lead_weights
```

---

## 3. ECG Data Handling

### 3.1 Data Preprocessing Pipeline

```python
# 1. Raw PTB-XL data: 21799 patients, 12 leads, 5000 samples @ 500 Hz
# 2. Bandpass filter: 0.5 - 40 Hz (removes baseline wander, high-freq noise)
# 3. Notch filter: 50 Hz (removes powerline interference)
# 4. Fixed-length windowing: 1000 samples (2 seconds @ 500 Hz)
#    - Window size: 1000 samples
#    - Stride: 500 samples (50% overlap)
# 5. Z-score normalization: (x - μ) / σ per lead
```

**Why 1000 samples (2 seconds)?**
- Captures 1-2 complete cardiac cycles (typical HR: 60-100 bpm)
- Sufficient to analyze P-QRS-T waveforms
- Balances temporal context and computational efficiency

### 3.2 Data Shapes Through the Model

| Stage | Shape | Description |
|-------|-------|-------------|
| Raw ECG | `(B, 1000, 12)` | Batch of normalized 2-second windows |
| BiLSTM Encoder Output | `(B, 1000, 256)` | Bidirectional hidden states at each timestep |
| Temporal Context | `(B, 256)` | Attention-weighted temporal representation |
| Lead Features | `(B, 12)` | Attention-weighted lead representation |
| Combined Features | `(B, 268)` | Concatenated temporal + lead features |
| Latent μ, σ | `(B, 32)` | Mean and log-variance of latent distribution |
| Latent z | `(B, 32)` | Sampled latent vector |
| Decoder Input | `(B, 1000, 128)` | Repeated latent projection |
| Decoder Output | `(B, 1000, 256)` | Reconstructed hidden states |
| Reconstruction | `(B, 1000, 12)` | Final 12-lead ECG reconstruction |

### 3.3 Dataset Statistics

**Training Set**: 7,611 samples (windowed from PTB-XL)
**Validation Set**: 1,903 samples
**Test Set**: 9,514 normal + 12,319 abnormal = 21,833 windows

**Class Distribution**:
- Normal: 9,514 windows (43.6%)
- Abnormal: 12,319 windows (56.4%)

---

## 4. Loss Function and Training

### 4.1 VAE Loss Function

The model optimizes a composite loss:

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}
$$

#### 4.1.1 Reconstruction Loss
$$
\mathcal{L}_{\text{recon}} = \frac{1}{B \cdot T \cdot L} \sum_{b=1}^{B} \sum_{t=1}^{T} \sum_{l=1}^{L} (x_{btl} - \hat{x}_{btl})^2
$$

Where:
- $B$ = batch size (64)
- $T$ = timesteps (1000)
- $L$ = leads (12)
- $x$ = input ECG
- $\hat{x}$ = reconstructed ECG

**Interpretation**: Mean Squared Error (MSE) penalizes deviation between input and reconstruction. Lower MSE → better fidelity to original waveform morphology.

#### 4.1.2 KL Divergence Loss
$$
\mathcal{L}_{\text{KL}} = -\frac{1}{2B \cdot D} \sum_{b=1}^{B} \sum_{d=1}^{D} (1 + \log(\sigma_d^2) - \mu_d^2 - \sigma_d^2)
$$

Where:
- $D$ = latent dimensions (32)
- $\mu_d$ = mean of latent dimension $d$
- $\sigma_d^2$ = variance of latent dimension $d$

**Interpretation**: Regularizes latent space to follow $\mathcal{N}(0, 1)$. Prevents overfitting and ensures smooth, continuous latent representation.

#### 4.1.3 Beta-Annealing Strategy

$$
\beta(t) = \beta_{\max} \cdot \min\left(1, \frac{2t \mod P}{P}\right)
$$

Where:
- $\beta_{\max} = 0.3$ (maximum beta weight)
- $P = 10$ epochs (cycle period)
- $t$ = current epoch

**Annealing Schedule**:
```
Epochs 0-5:   β = 0.0 → 0.3 (linear ramp-up)
Epochs 5-10:  β = 0.3 → 0.0 (linear ramp-down)
Epochs 10-15: β = 0.0 → 0.3 (cycle repeats)
```

**Purpose**: Cyclical annealing allows the model to:
1. **Focus on reconstruction early** (β=0): Learn to reproduce input accurately
2. **Regularize periodically** (β=0.3): Prevent latent space collapse
3. **Balance both objectives**: Achieve high-fidelity reconstruction with structured latent space

### 4.2 Training Configuration

**Optimizer**: Adam
- Learning rate: 0.0005 (initial)
- Betas: (0.9, 0.999)
- Weight decay: 1e-5

**Learning Rate Scheduler**: ReduceLROnPlateau
- Factor: 0.5 (halve LR when plateaued)
- Patience: 5 epochs
- Schedule:
  - Epochs 1-21: LR = 0.0005
  - Epochs 22-50: LR = 0.00025 (reduced after validation loss plateau)

**Gradient Clipping**: Max norm = 1.0

**Batch Size**: 64

**Epochs**: 50 (best model at epoch 50)

### 4.3 Training Dynamics

**Loss Progression**:

| Epoch | Train Loss | Val Loss | Train Recon | Val Recon | Train KL | Val KL |
|-------|------------|----------|-------------|-----------|----------|--------|
| 1 | 3818.21 | 1085.56 | 3736.96 | 1032.79 | 1354.22 | 879.56 |
| 10 | 179.43 | 154.70 | 164.90 | 140.18 | 48.44 | 48.41 |
| 20 | 161.16 | 146.22 | 140.00 | 128.74 | 47.55 | 47.21 |
| 30 | 143.36 | 134.24 | 134.31 | 113.46 | 45.48 | 44.73 |
| 40 | 145.43 | 131.87 | 131.21 | 112.60 | 39.31 | 47.44 |
| **50** | **139.29** | **111.35** | **123.41** | **104.87** | **43.73** | **44.94** |

**Key Observations**:
1. **Rapid initial convergence**: Loss drops 95% in first 10 epochs
2. **KL stabilization**: KL loss stabilizes around 45 after epoch 10 (successful regularization)
3. **Reconstruction improvement**: Validation recon loss decreases steadily to 104.87 (best model)
4. **LR reduction impact**: After epoch 22, LR reduction helps fine-tune reconstruction

---

## 5. Anomaly Score Computation

### 5.1 Reconstruction Error Metric

The anomaly score for each sample is computed as:

$$
\text{Anomaly Score}(x) = \frac{1}{T \cdot L} \sum_{t=1}^{T} \sum_{l=1}^{L} (x_{tl} - \hat{x}_{tl})^2
$$

**Implementation**:
```python
def compute_anomaly_score(x, model):
    x_recon, mu, logvar, _, _ = model(x)
    mse = F.mse_loss(x_recon, x, reduction='none')  # (B, 1000, 12)
    anomaly_score = mse.mean(dim=[1, 2])            # (B,)
    return anomaly_score
```

**Interpretation**:
- **Normal ECG**: Low reconstruction error (model trained predominantly on normal)
- **Abnormal ECG**: High reconstruction error (deviates from learned normal patterns)

### 5.2 Threshold Selection

**Method**: 95th percentile of reconstruction errors on validation set

```python
val_scores = []
for batch in val_loader:
    scores = compute_anomaly_score(batch, model)
    val_scores.extend(scores.cpu().numpy())

threshold = np.percentile(val_scores, 95)  # threshold = 0.045
```

**Classification Rule**:
```python
if anomaly_score > 0.045:
    prediction = "Abnormal"
else:
    prediction = "Normal"
```

### 5.3 Confusion Matrix Analysis

|  | Predicted Normal | Predicted Abnormal |
|--|------------------|-------------------|
| **Actual Normal** | TN = 6,956 (73.1%) | FP = 2,558 (26.9%) |
| **Actual Abnormal** | FN = 2,710 (22.0%) | TP = 9,609 (78.0%) |

**Derived Metrics**:
- **Precision** = TP / (TP + FP) = 9609 / (9609 + 2558) = **0.79**
- **Recall** = TP / (TP + FN) = 9609 / (9609 + 2710) = **0.78**
- **F1-Score** = 2 × (P × R) / (P + R) = **0.785**
- **Specificity** = TN / (TN + FP) = 6956 / 9514 = **0.731**

---

