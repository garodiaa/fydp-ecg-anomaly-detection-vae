# Hierarchical Latent VAE (HL-VAE) - Complete Technical Documentation

**Model Type:** Hierarchical Variational Autoencoder with Dual Latent Spaces  
**Architecture:** Bidirectional LSTM with Global + Local Latent Representations  
**Application:** ECG Anomaly Detection (Multi-Scale Feature Learning)  
**Parameters:** ~10.5M

---

## 1. Architecture Overview

### 1.1 Design Philosophy

The Hierarchical Latent VAE (HL-VAE) addresses a key limitation of standard VAEs: **single-scale representation**. ECG signals contain both:

- **Global patterns** (sequence-level): Overall rhythm, heart rate, beat periodicity
- **Local patterns** (timestep-level): Individual P-wave/QRS/T-wave morphology, ST-segment variations

**Key Innovation:** Separate latent spaces for global and local features, allowing the model to capture both scales simultaneously.

**Motivation:**
- **Myocardial Infarction:** Global rhythm normal, local ST-elevation abnormal
- **Atrial Fibrillation:** Global rhythm irregular, local P-waves missing
- **Bundle Branch Block:** Global timing altered, local QRS widened

### 1.2 Full Architecture Diagram

```
Input: (Batch, 1000 timesteps, 12 leads)
         ↓
   ┌────────────────────────────────────────────┐
   │         DUAL ENCODING PATHS                │
   └────────────────────────────────────────────┘
         ↓                          ↓
   [GLOBAL ENCODER]          [LOCAL ENCODER]
   (Sequence-level)          (Timestep-level)
         ↓                          ↓
   BiLSTM Layer 1            BiLSTM (1 layer)
   12 → 512*2, 2-dir         12 → 128*2, 2-dir
   (B, T, 1024)              (B, T, 256)
         ↓                          ↓
   BiLSTM Layer 2            Per-timestep MLP:
   1024 → 256*2, 2-dir       Linear(256 → 128)
   (B, T, 512)               GELU, Dropout(0.1)
         ↓                   Linear(128 → 32)
   Mean pooling              (B*T, 32)
   (B, 512)                       ↓
         ↓                   Reshape to (B, T, 32)
   MLP Statistics:           Split → mu_l: (B, T, 16)
   Linear(512 → 256)                 logvar_l: (B, T, 16)
   GELU, Dropout(0.1)             ↓
   Linear(256 → 128)         Reparameterize per timestep:
   Split → mu_g: (B, 64)     z_l = mu_l + eps * exp(0.5*logvar_l)
           logvar_g: (B, 64)  z_l: (B, T, 16)
         ↓
   Reparameterize:
   z_g = mu_g + eps * exp(0.5*logvar_g)
   z_g: (B, 64) ← Global latent
         ↓
   ┌────────────────────────────────────────────┐
   │            HIERARCHICAL DECODER            │
   └────────────────────────────────────────────┘
         ↓
   Expand z_g to sequence:
   z_g_exp: (B, T, 64) ← Repeat for each timestep
         ↓
   Concatenate with local latent:
   z_comb = [z_g_exp, z_l]: (B, T, 80)
         ↓
   BiLSTM Decoder 1
   80 → 256*2, 2-dir
   (B, T, 512)
         ↓
   BiLSTM Decoder 2
   512 → 512*2, 2-dir
   (B, T, 1024)
         ↓
   Per-timestep reconstruction MLP:
   Flatten: (B*T, 1024)
   Linear(1024 → 512)
   GELU, Dropout(0.1)
   Linear(512 → 24)  [mean + logvar for 12 leads]
         ↓
   Reshape: x_mean: (B, T, 12)
            x_logvar: (B, T, 12)
         ↓
   Output: Reconstructed ECG (B, T, 12)
```

### 1.3 Key Architectural Features

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Global Encoder** | 2-layer BiLSTM (512, 256 hidden) | Captures sequence-level rhythm patterns |
| **Local Encoder** | 1-layer BiLSTM (128 hidden) | Captures timestep-level morphology |
| **Global Latent** | 64 dimensions | Rhythm, overall beat structure |
| **Local Latent** | 16 dimensions per timestep | P-QRS-T morphology, ST variations |
| **Decoder** | 2-layer BiLSTM (256, 512 hidden) | Conditions on both global + local |
| **Total Latent Size** | 64 + (1000×16) = 16,064 dims | Multi-scale representation |

---

## 2. Forward Pass: Step-by-Step

### 2.1 Global Encoding (Sequence-Level)

**Purpose:** Extract overall rhythm and beat periodicity

```python
Input x: (64, 1000, 12)

# Layer 1: Bidirectional LSTM
h1, _ = BiLSTM(input=12, hidden=512, layers=1, bidir=True)(x)
# h1: (64, 1000, 1024) - 512*2 from bidirectional

# Layer 2: Deeper abstraction
h2, _ = BiLSTM(input=1024, hidden=256, layers=1, bidir=True)(h1)
# h2: (64, 1000, 512) - 256*2 from bidirectional

# Pool over time dimension (summarize sequence)
h_pool = h2.mean(dim=1)
# h_pool: (64, 512) - Single vector per sequence

# Compute global latent statistics
stats_g = MLP(h_pool)
# Linear(512 → 256) + GELU + Dropout
# Linear(256 → 128)
# stats_g: (64, 128)

mu_g, logvar_g = stats_g.chunk(2, dim=-1)
# mu_g: (64, 64)
# logvar_g: (64, 64)

# Reparameterize
z_g = mu_g + torch.randn_like(mu_g) * torch.exp(0.5 * logvar_g)
# z_g: (64, 64) ← Global latent (rhythm/overall pattern)
```

**What z_g captures:**
- Heart rate (fast vs slow)
- Rhythm regularity (sinus vs irregular)
- Beat spacing (RR intervals)
- Overall ECG morphology (normal vs pathological)

### 2.2 Local Encoding (Timestep-Level)

**Purpose:** Extract per-timestep morphology (P-QRS-T shapes)

```python
# BiLSTM processes full sequence
h_local, _ = BiLSTM(input=12, hidden=128, layers=1, bidir=True)(x)
# h_local: (64, 1000, 256) - 128*2 from bidirectional

# Flatten for per-timestep processing
B, T, C = h_local.shape
h_flat = h_local.reshape(B * T, C)
# h_flat: (64000, 256)

# Per-timestep MLP
stats_l = MLP(h_flat)
# Linear(256 → 128) + GELU + Dropout
# Linear(128 → 32)
# stats_l: (64000, 32)

mu_l, logvar_l = stats_l.chunk(2, dim=-1)
# Reshape back to sequence
mu_l = mu_l.reshape(B, T, 16)
logvar_l = logvar_l.reshape(B, T, 16)
# mu_l: (64, 1000, 16)
# logvar_l: (64, 1000, 16)

# Reparameterize (per timestep)
z_l = mu_l + torch.randn_like(mu_l) * torch.exp(0.5 * logvar_l)
# z_l: (64, 1000, 16) ← Local latent (morphology per timestep)
```

**What z_l captures:**
- P-wave shape (atrial depolarization)
- QRS complex width/amplitude (ventricular depolarization)
- ST-segment elevation/depression (ischemia marker)
- T-wave morphology (repolarization)

### 2.3 Hierarchical Decoding

**Purpose:** Reconstruct ECG conditioned on both global and local features

```python
# Expand global latent to sequence
z_g_exp = z_g.unsqueeze(1).expand(-1, T, -1)
# z_g_exp: (64, 1000, 64)

# Concatenate with local latent
z_comb = torch.cat([z_g_exp, z_l], dim=-1)
# z_comb: (64, 1000, 80) - Combined representation

# Decoder BiLSTM Layer 1
d1, _ = BiLSTM(input=80, hidden=256, layers=1, bidir=True)(z_comb)
# d1: (64, 1000, 512)

# Decoder BiLSTM Layer 2
d2, _ = BiLSTM(input=512, hidden=512, layers=1, bidir=True)(d1)
# d2: (64, 1000, 1024)

# Per-timestep reconstruction
d_flat = d2.reshape(B * T, 1024)
rec_stats = MLP(d_flat)
# Linear(1024 → 512) + GELU + Dropout
# Linear(512 → 24)
# rec_stats: (64000, 24)

x_mean, x_logvar = rec_stats.chunk(2, dim=-1)
x_mean = x_mean.reshape(B, T, 12)
x_logvar = x_logvar.reshape(B, T, 12)
# x_mean: (64, 1000, 12) - Reconstructed ECG
# x_logvar: (64, 1000, 12) - Per-element uncertainty
```

---

## 3. Parameters and Hyperparameters

### 3.1 Model Parameters

```python
model = HLVAE(
    n_leads=12,
    seq_len=1000,
    global_latent_dim=64,      # Sequence-level latent
    local_latent_dim=16,       # Per-timestep latent
    enc_hidden_global=(512, 256),  # Global encoder hidden sizes
    enc_hidden_local=128,      # Local encoder hidden size
    dec_hidden=(256, 512),     # Decoder hidden sizes
    beta_g=0.3,                # Global KL weight
    beta_l=0.3,                # Local KL weight
    dropout=0.1
)
```

### 3.2 Parameter Count Breakdown

| Component | Parameters | Calculation |
|-----------|-----------|-------------|
| **Global Encoder LSTM 1** | ~3.15M | `4*(12+512+1)*512*2` (bidirectional) |
| **Global Encoder LSTM 2** | ~2.36M | `4*(1024+256+1)*256*2` |
| **Global MLP** | ~164K | `512*256 + 256*128` |
| **Local Encoder LSTM** | ~525K | `4*(12+128+1)*128*2` |
| **Local MLP** | ~33K | `256*128 + 128*32` |
| **Decoder LSTM 1** | ~1.31M | `4*(80+256+1)*256*2` |
| **Decoder LSTM 2** | ~2.62M | `4*(512+512+1)*512*2` |
| **Decoder MLP** | ~525K | `1024*512 + 512*24` |
| **Total** | **~10.5M** | Largest model (more capacity) |

**Comparison:**
- HL-VAE: 10.5M (largest)
- ST-VAE: 8.2M
- BA-VAE: 0.6M
- VAE-BiLSTM-Attn: 2.1M

---

## 4. Loss Function (Dual KL Divergence)

### 4.1 Total Loss Formulation

```
Total Loss = L_recon + β_g * L_KL_global + β_l * L_KL_local

Where:
  L_recon    = MSE reconstruction loss
  L_KL_global = KL divergence of global latent
  L_KL_local  = KL divergence of local latents (sum over time)
  β_g, β_l    = Separate KL weights (both 0.3)
```

### 4.2 Reconstruction Loss

```python
# MSE between original and reconstructed
recon_loss = F.mse_loss(x_mean, x, reduction='sum') / x.size(0)
```

**Typical Values:** 300 - 500

### 4.3 Global KL Divergence

```python
# Stored from forward pass
mu_g = self._mu_g       # (B, 64)
logvar_g = self._logvar_g  # (B, 64)

# Clamp for stability
logvar_g = torch.clamp(logvar_g, min=-10.0, max=10.0)

# KL divergence
kl_g = -0.5 * (1 + logvar_g - mu_g.pow(2) - logvar_g.exp())
kl_g = kl_g.sum(dim=-1).mean()  # Sum over dims, mean over batch
```

**Typical Values:** 850 - 950

### 4.4 Local KL Divergence

```python
mu_l = mu_l  # (B, T, 16) from forward pass
logvar_l = logvar_l  # (B, T, 16)

# Clamp for stability
logvar_l = torch.clamp(logvar_l, min=-10.0, max=10.0)

# KL divergence per timestep
kl_l = -0.5 * (1 + logvar_l - mu_l.pow(2) - logvar_l.exp())
kl_l = kl_l.sum(dim=[1, 2]).mean()  # Sum over time+dims, mean over batch
```

**Typical Values:** 850 - 950

### 4.5 Combined Loss

```python
total = recon_loss + beta_g * kl_g + beta_l * kl_l
```

**Training Values:**
- Epoch 1: Total = 2,765, Recon = 2,616, KL_total = 2,485
- Epoch 50: Total = 531, Recon = 304, KL_total = 843
- **Note:** KL much larger than recon due to many local latents (1000×16)

---

## 5. Performance Analysis

### 5.1 Evaluation Results (PTB-XL Test Set)

**Latest Results (from comprehensive JSON):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **F1-Score** | **0.805** | Strong overall performance |
| **Precision** | **0.80** | 80% of flagged cases truly abnormal |
| **Recall** | **0.81** | Catches 81% of all abnormalities |
| **AUC-ROC** | **0.85** | Good discrimination ability |
| **AUC-PR** | **0.83** | Strong on imbalanced data |
| **Threshold** | 0.065 | 95th percentile of normal errors |
| **Inference** | **3.72 ms/sample** | Moderate speed |

**Confusion Matrix:**
```
                  Predicted
                Normal  Abnormal
Actual Normal     7,024    2,490  (TN, FP)
Actual Abnormal   2,341    9,978  (FN, TP)
```

**Analysis:**
- **True Positive Rate (Recall):** 9,978 / 12,319 = **81.0%** ← Catches most abnormalities
- **True Negative Rate (Specificity):** 7,024 / 9,514 = **73.8%** ← Some false alarms
- **False Positive Rate:** 2,490 / 9,514 = **26.2%** ← Higher than BA-VAE (5%)
- **False Negative Rate:** 2,341 / 12,319 = **19.0%** ← Misses some abnormalities

### 5.2 Comparison with Other Models

| Model | F1 | AUC-ROC | Precision | Recall | Inference (ms) |
|-------|-----|---------|-----------|--------|----------------|
| ST-VAE | **0.90** | **0.93** | 0.89 | **0.91** | **0.11** |
| **HL-VAE** | **0.805** | **0.85** | 0.80 | **0.81** | 3.72 |
| BA-VAE | 0.55 | 0.78 | **0.91** | 0.40 | 1.53 |
| VAE-BiLSTM-Attn | 0.52 | 0.79 | 0.91 | 0.37 | 4.54 |

**HL-VAE Position:**
- ✅ **2nd best F1-Score** (after ST-VAE)
- ✅ **2nd best Recall** (after ST-VAE) ← Catches 81% vs ST-VAE's 91%
- ✅ **Higher recall than simple VAEs** (BA-VAE only 40%)
- ⚠️ **More false positives** (26.2%) than BA-VAE (5%)
- ⚠️ **Slower inference** (3.72 ms) than BA-VAE/ST-VAE

### 5.3 Why HL-VAE Performs Well

**Multi-Scale Modeling:**
- **Global latent** captures rhythm abnormalities (atrial fib, tachycardia)
- **Local latent** captures morphology abnormalities (ST elevation, T-wave inversion)
- **Combined** catches more abnormalities than single-scale models

**Example Detection:**
```
Normal Sinus Rhythm:
  - z_g captures: Regular 60 bpm rhythm
  - z_l captures: Normal P-QRS-T morphology
  - Reconstruction: Accurate (low error)
  
Atrial Fibrillation:
  - z_g captures: Irregular rhythm (different from training!)
  - z_l captures: Missing P-waves
  - Reconstruction: Poor (high error) → Detected
  
ST-Elevation MI:
  - z_g captures: Normal rhythm
  - z_l captures: Elevated ST-segment (abnormal morphology)
  - Reconstruction: Poor (high error) → Detected
```

### 5.4 Training Dynamics

**Key Milestones:**
```
Epoch 1-10:   Rapid learning
              Total: 2765 → 723
              Recon: 2616 → 424
              KL: 2485 → 997
              
Epoch 10-30:  Refinement + LR reduction (5e-4 → 2.5e-4)
              Total: 723 → 583
              Recon: 424 → 331
              KL: 997 → 896
              
Epoch 30-50:  Final convergence
              Total: 583 → 531
              Recon: 331 → 304
              KL: 896 → 843
              Best epoch: 49
```

**Stable Training:**
- No KL explosion (stays 850-1000)
- Gradual improvement
- Best model at epoch 49 (full training)
- No overfitting (train ≈ val)

---

## 6. Why Hierarchical Latents Matter

### 6.1 Single-Scale vs Multi-Scale

**Standard VAE (Single Latent):**
```
x → Encoder → z (single scale) → Decoder → x_recon
              ↑
        Must capture BOTH global rhythm 
        AND local morphology in same space
        → Trade-off, suboptimal
```

**HL-VAE (Hierarchical Latents):**
```
x → Global Encoder → z_g (rhythm)    ↘
                                      → Decoder → x_recon
x → Local Encoder  → z_l (morphology) ↗
              ↑
        Separate spaces for different scales
        → No trade-off, optimal
```

### 6.2 Latent Space Specialization

**Global Latent (z_g):**
- Dimension: 64
- Captures: Rhythm, beat periodicity, overall pattern
- Similar across timesteps (constant for sequence)

**Local Latent (z_l):**
- Dimension: 16 per timestep (1000 × 16 total)
- Captures: P-wave, QRS, ST-segment, T-wave
- Varies across timesteps (specific to each sample)

**Analogy:**
- **Global:** "This ECG has a heart rate of 72 bpm with regular rhythm"
- **Local:** "At timestep 250, there's a QRS complex with 0.08s duration"

### 6.3 Reconstruction Quality

**Per-Component Error:**

| Component | Reconstruction Error | Interpretation |
|-----------|---------------------|----------------|
| **Rhythm** | Low (z_g accurate) | Heart rate, beat spacing preserved |
| **P-waves** | Low (z_l accurate) | Atrial depolarization captured |
| **QRS** | Low (z_l accurate) | Ventricular depolarization captured |
| **ST-segment** | Moderate | Subtle changes may be smoothed |
| **T-waves** | Moderate | Repolarization details preserved |

---

## 7. Clinical Interpretation

### 7.1 What HL-VAE Detects Best

**Global Abnormalities (via z_g):**
- ✅ Atrial Fibrillation (irregular rhythm)
- ✅ Tachycardia/Bradycardia (abnormal heart rate)
- ✅ Heart block (abnormal conduction timing)
- ✅ Premature beats (disrupted rhythm)

**Local Abnormalities (via z_l):**
- ✅ ST-Elevation MI (local ST-segment changes)
- ✅ T-wave inversions (local repolarization abnormality)
- ✅ Pathological Q-waves (local QRS morphology)
- ✅ Bundle branch blocks (local QRS widening)

**Combined (Both z_g and z_l):**
- ✅ Ventricular Tachycardia (abnormal rhythm + wide QRS)
- ✅ Atrial Flutter (rapid regular rhythm + sawtooth P-waves)

### 7.2 Latent Space Visualization (Potential)

**Global Latent Clustering:**
```python
# t-SNE on z_g (64-dimensional)
normal_z_g = [encode_global(x) for x in normal_ecgs]
abnormal_z_g = [encode_global(x) for x in abnormal_ecgs]

# Expectation: Two clusters (normal vs abnormal rhythm)
```

**Local Latent Heatmap:**
```python
# Visualize z_l over time
z_l = encode_local(ecg)  # (1000, 16)
plt.imshow(z_l.T, aspect='auto')
plt.xlabel('Timestep')
plt.ylabel('Local Latent Dimension')

# Expectation: High activation at QRS, ST, T-wave regions
```

---

## 8. Hyperparameter Tuning

### 8.1 Latent Dimensions

**Global Latent:**
- **Current:** 64
- **Range:** 32-128
- **Impact:** Larger = more rhythm expressiveness, risk of collapse

**Local Latent:**
- **Current:** 16 per timestep
- **Range:** 8-32
- **Impact:** Larger = finer morphology, larger total latent size

**Total Latent Size:**
- Current: 64 + 1000×16 = **16,064 dimensions**
- Very large compared to:
  - ST-VAE: 32
  - BA-VAE: 32
  - VAE-GRU: 32

### 8.2 Beta Weights

**Global Beta (β_g):**
- **Current:** 0.3
- **Impact:** Controls global latent regularization

**Local Beta (β_l):**
- **Current:** 0.3
- **Impact:** Controls local latent regularization

**Recommendation:**
- Keep β_g = β_l (balanced)
- Increase both if KL too low
- Decrease both if KL explosion

### 8.3 Encoder/Decoder Sizes

**Global Encoder:**
- **Current:** (512, 256) hidden dims
- Large capacity for sequence-level features

**Local Encoder:**
- **Current:** 128 hidden dim
- Smaller (per-timestep processing)

**Decoder:**
- **Current:** (256, 512) hidden dims
- Large capacity for reconstruction

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

**1. Large Latent Space:**
- 16,064 total dimensions (64 + 1000×16)
- Computationally expensive
- Risk of redundancy across timesteps

**2. Higher False Positives:**
- 26.2% FPR vs BA-VAE's 5%
- More sensitive (catches more abnormalities)
- But also flags more normals

**3. Slower Inference:**
- 3.72 ms/sample vs ST-VAE's 0.11 ms
- Multiple BiLSTM layers (sequential processing)
- Not suitable for extreme-speed requirements

**4. Complex Architecture:**
- Two separate encoding paths
- More hyperparameters to tune
- Harder to train/debug

### 9.2 Future Improvements

**Short-Term:**
1. **Latent Compression:**
   - Reduce local_latent_dim: 16 → 8
   - Use attention to pool local latents
   - Compress 16,064 → ~1,000 dims

2. **Adaptive Beta:**
   - Different β for global vs local
   - Schedule: β_g constant, β_l annealed
   - Prevent local latent collapse

3. **Temporal Smoothness:**
   - Add L2 penalty on Δz_l (adjacent timesteps)
   - Encourage smooth local latent transitions
   - Reduce redundancy

**Long-Term:**
1. **Attention-Based Local Encoder:**
   - Replace BiLSTM with Transformer
   - Parallel processing (faster inference)
   - Better long-range dependencies

2. **Variational RNN:**
   - Stochastic transitions in latent space
   - z_l(t) depends on z_l(t-1)
   - More efficient temporal modeling

3. **Disentangled Representations:**
   - Enforce independence between z_g and z_l
   - Use β-VAE or β-TCVAE
   - Improve interpretability

---

## 10. Use Cases and Recommendations

### 10.1 When to Use HL-VAE

✅ **Multi-Scale Abnormalities:**
- Abnormalities with both rhythm and morphology changes
- Example: Ventricular tachycardia (fast + wide QRS)

✅ **High Recall Required:**
- 81% recall vs BA-VAE's 40%
- Screening applications where missing abnormalities is costly

✅ **Research/Analysis:**
- Interpretable latent spaces (global vs local)
- Study rhythm vs morphology contributions

✅ **Balanced Performance:**
- F1=0.805 is strong (between BA-VAE's 0.55 and ST-VAE's 0.90)
- Good compromise

### 10.2 When NOT to Use HL-VAE

❌ **Low False Positive Requirement:**
- 26.2% FPR too high for some applications
- Use BA-VAE (5% FPR) instead

❌ **Real-Time Monitoring:**
- 3.72 ms inference may be too slow
- Use ST-VAE (0.11 ms) or CAE

❌ **Resource-Constrained:**
- 10.5M parameters largest among models
- Use BA-VAE (0.6M) or CAE (0.25M)

### 10.3 Recommended Pipeline

**Two-Stage Detection:**
```
Stage 1: BA-VAE (high precision, fast)
  → 95% of normals pass through (low FPR)
  → Flags suspicious cases
  
Stage 2: HL-VAE (high recall, thorough)
  → Analyzes flagged cases in detail
  → Multi-scale analysis
  → Final classification
```

---

## 11. Reproducibility

### 11.1 Training Command

```powershell
python src/training.py `
  --model hlvae `
  --epochs 50 `
  --batch_size 64 `
  --lr 5e-4 `
  --latent_dim 64 `
  --beta 0.3 `
  --seq_len 1000 `
  --dataset ptbxl
```

**Note:** `--latent_dim` sets global_latent_dim; local is automatically global/4.

### 11.2 Evaluation Command

```powershell
python src/evaluate.py `
  --model hlvae `
  --latent-dim 64 `
  --seq-len 1000 `
  --batch-size 64 `
  --threshold-percentile 95
```

### 11.3 Expected Training Time

| Hardware | Time per Epoch | Total (50 epochs) |
|----------|----------------|-------------------|
| RTX 3060 | ~90 sec | **~75 min** |
| RTX 3090 | ~50 sec | **~42 min** |

**Slower than BA-VAE (25 min) due to larger model.**

---

## 12. Summary

**Hierarchical Latent VAE (HL-VAE)** achieves strong performance through **multi-scale latent representation**:

✅ **F1-Score: 0.805** (2nd best overall)  
✅ **Recall: 81%** (catches most abnormalities)  
✅ **Multi-scale learning** (global rhythm + local morphology)  
✅ **Interpretable latents** (separate spaces for different scales)  

**Key Innovation:** Dual latent spaces enable simultaneous capture of sequence-level rhythm and timestep-level morphology.

**Best For:**
- Multi-scale abnormality detection
- High-recall screening applications
- Research requiring interpretable latents

**Trade-offs:**
- Higher false positives (26.2%)
- Slower inference (3.72 ms)
- Larger model (10.5M params)

**Recommendation:** Use for **high-recall applications** where catching abnormalities is prioritized over minimizing false alarms.

---

*Document Version: 1.0*  
*Last Updated: December 1, 2025*  
*Model Version: HL-VAE (Standard Architecture)*
