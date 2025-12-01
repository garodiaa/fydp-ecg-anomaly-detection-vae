# Study Documentation: ECG Anomaly Detection with VAE-based Models

## 1. Dataset Overview

### 1.1 Dataset Source: PTB-XL

This study utilizes the **PTB-XL database**, a large publicly available electrocardiography dataset containing 21,837 clinical 12-lead ECG records from 18,885 patients. PTB-XL is one of the largest ECG datasets available for research purposes.

**Key Dataset Characteristics:**
- **Source**: Physikalisch-Technische Bundesanstalt (PTB), Germany
- **Total Records**: 21,837 ECG recordings
- **Leads**: 12-lead ECG (I, II, III, aVR, aVL, aVF, V1-V6)
- **Duration**: 10 seconds per recording
- **Sampling Rates**: 
  - High Resolution (HR): 500 Hz
  - Low Resolution (LR): 100 Hz
- **Annotations**: Diagnostic labels using SCP-ECG codes
- **Diagnostic Categories**: NORM (normal), MI (myocardial infarction), HYP (hypertrophy), STTC (ST-T changes), CD (conduction disturbances), and others

### 1.2 Normal ECG Dataset for Reconstruction Training

For training the VAE-based reconstruction models, we use **only normal ECG recordings** from PTB-XL.

**Normal Dataset Specifications:**
- **Total Normal ECGs**: 9,514 recordings
- **Selection Criteria**: ECGs labeled with diagnostic superclass "NORM" (normal ECG)
- **Filtering Process**: 
  - Parsed `scp_codes` from `ptbxl_database.csv`
  - Extracted diagnostic superclass using `scp_statements.csv` mapping
  - Selected records where `diagnostic_superclass` contains "NORM"
  - Excluded all abnormal conditions (MI, HYP, STTC, CD, etc.)
  
**Rationale**: VAE models are trained exclusively on normal ECGs to learn the distribution and patterns of healthy cardiac signals. During inference, abnormal ECGs will produce higher reconstruction errors, enabling anomaly detection.

**Dataset Files**:
- Stored in: `data/processed/ptbxl/`
- Format: NumPy arrays (`.npy`)
- Shape: `(12, T)` where T ≈ 5000 samples (10 seconds at 500 Hz)
- Data Type: `float32`

### 1.3 Abnormal ECG Dataset for Evaluation

For evaluating anomaly detection performance, we created a separate dataset of **abnormal ECGs** with specific pathologies.

**Abnormal Dataset Specifications:**
- **Total Abnormal ECGs**: Varies by diagnostic class
- **Diagnostic Classes Included**:
  - **MI (Myocardial Infarction)**: Includes IMI (inferior), AMI (anterior), LMI (lateral), ASMI (anteroseptal), ALMI (anterolateral), PMI (posterior), ILMI (inferolateral), IPMI (inferoposterior), IPLMI (inferoposterolateral)
  - **HYP (Hypertrophy)**: LVH (left ventricular hypertrophy), RVH (right ventricular hypertrophy), LAO/LAE (left atrial overload/enlargement), RAO/RAE (right atrial overload/enlargement), SEHYP (septal hypertrophy)
  - **STTC (ST-T Changes)**: Non-specific ST-T changes, ischemic changes, digitalis effects, long QT
  - **CD (Conduction Disturbances)**: CRBBB (complete right bundle branch block), CLBBB (complete left bundle branch block), IRBBB/ILBBB (incomplete BBB), LAFB/LPFB (fascicular blocks), AVB (AV blocks), WPW (Wolff-Parkinson-White)

**Selection Criteria**: 
- ECGs with at least one diagnostic code belonging to MI, HYP, STTC, or CD superclasses
- Excluded normal ECGs (NORM class)

**Dataset Files**:
- Stored in: `data/processed/ptbxl_abnormal/`
- Format: NumPy arrays (`.npy`)
- Shape: `(T, 12)` where T ≈ 5000 samples
- Metadata: `abnormal_metadata.csv` and `diagnostic_summary.txt`

**Purpose**: These abnormal ECGs serve as the positive class for anomaly detection evaluation, allowing us to compute metrics like AUROC, AUPRC, sensitivity, and specificity.

---

## 2. Data Preprocessing Pipeline

The data preprocessing pipeline ensures consistent, high-quality ECG signals suitable for deep learning model training. All ECG recordings undergo identical preprocessing steps regardless of whether they are normal or abnormal.

### 2.1 Signal Filtering

#### 2.1.1 Bandpass Filtering (0.5 - 40 Hz)

**Purpose**: Remove baseline wander and high-frequency noise while preserving diagnostic ECG components.

**Implementation**:
```python
def butter_bandpass(signal, lowcut=0.5, highcut=40.0, fs=500, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)
```

**Technical Details**:
- **Filter Type**: Butterworth bandpass filter (4th order)
- **Low Cutoff**: 0.5 Hz (removes baseline drift and motion artifacts)
- **High Cutoff**: 40 Hz (removes muscle noise and powerline interference harmonics)
- **Zero-Phase Filtering**: Uses `filtfilt()` for forward-backward filtering to eliminate phase distortion
- **Application**: Applied independently to each of the 12 leads

**Rationale**:
- P-wave: 0.5-3 Hz
- QRS complex: 10-40 Hz (main diagnostic information)
- T-wave: 1-7 Hz
- Removes frequencies outside the diagnostic range

#### 2.1.2 Notch Filtering (50 Hz / 60 Hz)

**Purpose**: Remove powerline interference.

**Implementation**:
```python
def notch_filter(signal, freq=50.0, q=30, fs=500):
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, signal, axis=0)
```

**Technical Details**:
- **Filter Type**: IIR notch filter
- **Frequency**: 50 Hz (Europe) or 60 Hz (North America) - PTB-XL uses 50 Hz
- **Quality Factor (Q)**: 30 (narrow notch to minimize impact on adjacent frequencies)
- **Zero-Phase Filtering**: Uses `filtfilt()` for bidirectional filtering
- **Application**: Applied to each lead independently

**Rationale**: Powerline interference is a common artifact in ECG recordings that can mask diagnostic features.

### 2.2 Normalization

#### 2.2.1 Z-Score Normalization (Per-Lead)

**Purpose**: Standardize signal amplitudes across different leads and recordings for stable neural network training.

**Implementation**:
```python
def zscore(signal):
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0) + 1e-8
    return (signal - mean) / std
```

**Technical Details**:
- **Computation**: For each lead, subtract mean and divide by standard deviation
- **Per-Lead Normalization**: Applied independently to each of the 12 leads
- **Epsilon Term**: 1e-8 added to prevent division by zero for flat signals
- **Result**: Each lead has zero mean (μ = 0) and unit variance (σ = 1)

**Mathematical Formula**:
$$
x_{\text{normalized}} = \frac{x - \mu}{\sigma + \epsilon}
$$

where:
- $x$ is the original signal
- $\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$ (mean)
- $\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}$ (standard deviation)
- $\epsilon = 10^{-8}$ (small constant for numerical stability)

**Rationale**:
- Makes training more stable by ensuring consistent input ranges
- Prevents any single lead from dominating the loss function
- Facilitates convergence of gradient-based optimization
- Standard practice for time-series neural networks

#### 2.2.2 Online Z-Score Normalization in DataLoader

**Additional Normalization**: During model training, an additional z-score normalization is applied per-sample in the `ECGDataset` class.

```python
# In ECGDataset.__getitem__()
mean = sig.mean(dim=0, keepdim=True)
std = sig.std(dim=0, keepdim=True) + 1e-8
sig = (sig - mean) / std
```

**Purpose**: 
- Ensures each training sample has consistent statistics
- Accounts for any residual variations after preprocessing
- Critical for VAE training stability

### 2.3 Signal Length Standardization

**Fixed Length**: All ECG signals are standardized to **5000 samples**.

**Implementation**:
```python
sample_length = 5000
if L >= sample_length:
    sig = sig[:sample_length]  # Truncate
else:
    pad = torch.zeros(sample_length - L, C, dtype=sig.dtype)
    sig = torch.cat([sig, pad], dim=0)  # Zero-pad
```

**Technical Details**:
- **Original Length**: ~5000 samples (10 seconds at 500 Hz)
- **Truncation**: If signal exceeds 5000 samples, truncate to exactly 5000
- **Padding**: If signal is shorter, zero-pad to 5000 samples
- **Result**: All signals have shape `(5000, 12)`

**Rationale**: 
- Neural networks require fixed input dimensions
- Ensures batch processing compatibility
- 10 seconds captures complete cardiac cycles (typically 6-10 heartbeats at rest)

### 2.4 Data Quality Checks

#### 2.4.1 NaN and Inf Handling

**Implementation**:
```python
arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
```

**Purpose**: Replace any invalid values with zeros to prevent training failures.

#### 2.4.2 Shape Validation

**Checks**:
- Verify 12 leads present
- Ensure 2D array structure
- Validate temporal dimension

**Action**: Skip files that don't meet criteria and log warnings.

### 2.5 Data Format and Storage

**File Format**: NumPy binary (`.npy`)
- **Advantages**: Fast loading, compact storage, preserves precision
- **Data Type**: `float32` (reduces memory usage vs. `float64`)

**Shape Conventions**:
- **After Preprocessing**: `(12, T)` - channels first
- **In DataLoader**: `(T, 12)` - time-first for sequence models
- **Automatic Transposition**: DataLoader handles shape conversion

**Directory Structure**:
```
data/
├── processed/
│   ├── ptbxl/              # Normal ECGs (9,514 files)
│   │   ├── 00001_hr.npy
│   │   ├── 00002_hr.npy
│   │   └── ...
│   └── ptbxl_abnormal/     # Abnormal ECGs
│       ├── 00100.npy
│       ├── 00101.npy
│       ├── abnormal_metadata.csv
│       └── diagnostic_summary.txt
```

---

## 3. Preprocessing Scripts

### 3.1 Main Preprocessing Script: `pre-process.py`

**Location**: `src/data_processing/pre-process.py`

**Usage**:
```bash
# Process PTB-XL normal ECGs
python src/data_processing/pre-process.py \
    --dataset ptbxl \
    --input-dir ptbxl \
    --output-dir data/processed \
    --resolution auto

# Process with NaN cleaning
python src/data_processing/pre-process.py \
    --dataset ptbxl \
    --clean-nans
```

**Features**:
- Reads WFDB format ECG records
- Applies full preprocessing pipeline
- Saves processed signals as NumPy arrays
- Optional NaN detection and removal
- Progress tracking with tqdm

### 3.2 Abnormal Dataset Creation: `create_abnormal_dataset.py`

**Location**: `src/data_processing/create_abnormal_dataset.py`

**Usage**:
```bash
python src/data_processing/create_abnormal_dataset.py \
    --input-dir ptbxl \
    --output-dir data/processed/ptbxl_abnormal \
    --abnormal-classes MI HYP STTC CD \
    --resolution auto
```

**Features**:
- Filters ECGs by diagnostic superclass
- Applies identical preprocessing as normal ECGs
- Generates metadata and diagnostic summaries
- Tracks class distribution

### 3.3 Normalization Validation: `normalization_validation.py`

**Purpose**: Verify z-score normalization was applied correctly.

**Checks**:
- Mean ≈ 0 (within tolerance)
- Standard deviation ≈ 1 (within tolerance)
- No NaN or Inf values
- Correct shape

---

## 4. Data Loading for Model Training

### 4.1 ECGDataset Class

**Location**: `src/data_processing/dataset.py`

**Key Features**:
```python
class ECGDataset(Dataset):
    def __init__(self, ptbxl_dir, sample_length=5000):
        # Load all .npy files from directory
        
    def __getitem__(self, idx):
        # Load signal
        # Transpose if needed
        # Truncate/pad to sample_length
        # Apply online z-score normalization
        # Return tensor of shape (5000, 12)
```

**Usage in Training**:
```python
from torch.utils.data import DataLoader

train_dataset = ECGDataset(ptbxl_dir='data/processed/ptbxl')
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # GPU optimization
)
```

### 4.2 Train-Validation Split

**Split Ratio**: 80% training, 20% validation (default in `training.py`)

**Implementation**:
```python
from torch.utils.data import random_split

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
```

**Result**:
- **Training Set**: ~7,611 normal ECGs
- **Validation Set**: ~1,903 normal ECGs

**Random Seed**: Set for reproducibility

---

## 5. Preprocessing Quality Assurance

### 5.1 Signal Quality Metrics

**Checks Performed**:
1. **Baseline Drift Removed**: Verified by visual inspection
2. **Frequency Content**: Confirmed to be within 0.5-40 Hz range
3. **Normalization**: Mean ≈ 0, Std ≈ 1 for each lead
4. **No Artifacts**: NaN, Inf, or extreme outliers removed

### 5.2 Visual Validation

**Tools**: `visualization_vae.py`, `visualization_cae.py`

**Visualizations**:
- Raw vs. preprocessed signal comparison
- Power spectral density plots
- Lead-wise distribution histograms
- Reconstruction quality assessment

---

## 6. Summary of Data Preprocessing Pipeline

**Complete Pipeline**:
```
Raw WFDB Records (PTB-XL)
    ↓
1. Load 12-lead ECG signal
    ↓
2. Bandpass Filter (0.5-40 Hz, Butterworth, order=4)
    ↓
3. Notch Filter (50 Hz, Q=30)
    ↓
4. Z-Score Normalization (per-lead)
    ↓
5. Save as .npy (float32, shape: 12 × T)
    ↓
6. DataLoader: Transpose to (T × 12), truncate/pad to 5000 samples
    ↓
7. Online Z-Score Normalization (per-sample)
    ↓
8. Batch creation for model training
    ↓
Ready for VAE Training
```

**Final Dataset Statistics**:
- **Normal ECGs**: 9,514 recordings
- **Abnormal ECGs**: Varies by diagnostic class
- **Input Shape**: (5000, 12) - 10 seconds, 12 leads
- **Data Type**: float32 Tensor
- **Normalization**: Zero mean, unit variance per lead
- **Frequency Range**: 0.5-40 Hz (diagnostic ECG range)

---

## 7. Reproducibility

**Preprocessing Parameters**:
- Bandpass: 0.5-40 Hz (Butterworth, order=4)
- Notch: 50 Hz (Q=30)
- Normalization: Z-score (per-lead, ε=1e-8)
- Sample Length: 5000 samples
- Resolution: Auto-select (prefer 500 Hz)

**Random Seed**: Set in training scripts for reproducible train-val splits.

**Dependencies**:
- numpy
- scipy (signal processing)
- wfdb (ECG record reading)
- torch (data loading)
- pandas (metadata handling)

All preprocessing steps are fully documented and reproducible using the provided scripts in `src/data_processing/`.

---

## 8. MODEL ARCHITECTURES

This study evaluates nine distinct deep learning architectures for ECG anomaly detection, spanning from traditional convolutional approaches to state-of-the-art diffusion models. All models follow a reconstruction-based anomaly detection paradigm where normal ECGs are reconstructed with low error while abnormal ECGs exhibit higher reconstruction errors.

### 8.1 Convolutional Autoencoder (CAE)

The Convolutional Autoencoder represents the baseline architecture for this study, implementing a straightforward encoder-decoder structure adapted from the CAE-M model architecture.

**Architecture Philosophy**: CAE uses hierarchical convolutional layers to progressively compress the ECG signal into a bottleneck representation, then reconstructs it through symmetric transpose convolutions. The model operates purely in the spatial domain without explicit temporal modeling or attention mechanisms.

**Encoder Design**: The encoder consists of three convolutional blocks that downsample the input ECG from twelve leads across one thousand timesteps. Each block applies strided convolutions with kernel sizes decreasing from seven to three, paired with ReLU activations. The channel progression follows a standard pattern: input channels (twelve leads) expand to thirty-two, then sixty-four, and finally one hundred twenty-eight feature maps. Downsampling occurs through stride-two convolutions, reducing the temporal dimension by a factor of eight.

**Decoder Design**: The decoder mirrors the encoder architecture using transpose convolutions. Three blocks progressively upsample the compressed representation back to the original dimensions, with channel counts reversing from one hundred twenty-eight back to twelve. The final layer outputs the reconstructed twelve-lead ECG signal at the original one thousand sample length.

**Key Characteristics**: The model contains approximately two hundred fifty thousand parameters, making it the most lightweight architecture in this study. Training uses standard mean squared error loss without variational components or attention mechanisms. The simplicity enables fast training and inference but limits the model's ability to capture complex temporal dependencies and multi-scale patterns.

**Strengths**: Computationally efficient, fast inference, simple to train and deploy, serves as an effective baseline for comparison, and requires minimal hyperparameter tuning.

**Limitations**: No explicit temporal modeling beyond local convolutional receptive fields, lacks attention mechanisms for focusing on important regions, cannot capture long-range dependencies in ECG signals, and may struggle with subtle morphological variations.

### 8.2 Variational Autoencoder with GRU (VAE-GRU)

The VAE-GRU model introduces variational inference and recurrent processing to capture temporal dynamics in ECG signals, combining the probabilistic framework of Variational Autoencoders with the sequential processing capabilities of Gated Recurrent Units.

**Architecture Philosophy**: This model treats ECG analysis as a sequential prediction problem where temporal dependencies are crucial. The variational framework provides a probabilistic latent representation that captures uncertainty in the data, while GRU layers model the temporal evolution of cardiac signals.

**Encoder Architecture**: The encoder uses a multi-layer GRU with two hundred fifty-six hidden units to process the input sequence. The GRU operates on each timestep's twelve-lead measurements, accumulating context across the one thousand timestep sequence. The hidden states from the GRU are processed through a fully connected network with GELU activations and dropout for regularization. This network outputs both mean and log-variance parameters for a thirty-two dimensional latent distribution. Self-attention with four heads is applied to the GRU outputs before latent space projection, allowing the model to weight important temporal regions.

**Decoder Architecture**: The decoder receives samples from the latent distribution and generates reconstructions through a symmetric fully connected network. The decoder outputs both reconstruction mean and log-variance for each timestep and lead, enabling probabilistic reconstruction. The architecture processes each timestep independently after sampling from the latent space, reconstructing the full sequence.

**Loss Function**: The model optimizes a combination of reconstruction loss and KL divergence. Reconstruction loss uses mean squared error between the predicted mean and actual ECG values. KL divergence regularizes the latent space to approximate a standard Gaussian distribution, weighted by a beta parameter of zero point zero one. This beta-VAE formulation encourages disentangled representations.

**Key Characteristics**: The model contains approximately two hundred fifty thousand trainable parameters. It employs dropout regularization at zero point one, uses Xavier initialization for weights, and processes sequences in a recurrent manner enabling online inference. The attention mechanism provides interpretability through attention weights showing which temporal regions influence predictions.

**Strengths**: Captures temporal dependencies through recurrent processing, provides probabilistic uncertainty estimates, enables online sequential processing, includes attention for interpretability, and generates disentangled latent representations.

**Limitations**: Sequential processing is slower than parallel architectures, GRU may struggle with very long sequences despite gating mechanisms, single-layer attention is less powerful than multi-head cross-attention, and the relatively small latent dimension may limit representational capacity.

### 8.3 Variational Autoencoder with Bidirectional LSTM and Attention (VAE-BiLSTM-Attention)

This architecture significantly enhances the VAE-GRU approach by incorporating bidirectional processing, deeper networks, and sophisticated multi-head attention mechanisms inspired by the OmniAnomaly model from KDD 2019.

**Architecture Philosophy**: The model recognizes that ECG interpretation benefits from both past and future context at each timestep. Bidirectional processing allows the model to see complete cardiac cycles, while multi-head attention enables focusing on specific morphological features like P-waves, QRS complexes, and T-waves simultaneously.

**Encoder Architecture**: The encoder employs a two-layer bidirectional LSTM architecture with hidden dimensions of five hundred twelve and two hundred fifty-six respectively. Bidirectional processing means each timestep receives information from both preceding and following timesteps, doubling the effective hidden dimension. Input noise injection with standard deviation zero point zero one provides regularization. The encoder outputs per-timestep latent parameters (mean and log-variance) rather than a single global representation, enabling local anomaly detection. Each timestep produces a thirty-two dimensional latent code.

**Attention Mechanisms**: The model implements two complementary attention mechanisms. Multi-head attention with eight heads operates on temporal sequences, allowing the model to attend to different aspects of the ECG signal simultaneously (such as rhythm, morphology, and conduction patterns). Lead-wise attention with four heads processes each lead independently before fusion, recognizing that different ECG leads capture different cardiac electrical perspectives. The attention outputs are combined with the latent codes before decoding.

**Decoder Architecture**: The decoder uses a two-layer bidirectional LSTM architecture mirroring the encoder but with dimensions two hundred fifty-six and five hundred twelve. It takes the attention-weighted latent codes and reconstructs the original twelve-lead ECG. The final layer outputs both mean and log-variance for each timestep and lead, enabling uncertainty quantification in reconstruction.

**Loss Function**: Similar to VAE-GRU, the loss combines mean squared error reconstruction loss with KL divergence. However, KL is computed per-timestep rather than globally, allowing the model to identify specific temporal regions with high uncertainty. Beta weighting of zero point zero one prevents posterior collapse while encouraging meaningful latent representations.

**Key Characteristics**: The model contains approximately ten million parameters, making it one of the larger architectures. Training requires careful hyperparameter tuning, particularly for the beta parameter and learning rate. The bidirectional LSTMs and dual attention mechanisms provide rich representational capacity but increase computational cost.

**Strengths**: Bidirectional processing captures full temporal context, multiple attention heads learn complementary features, per-timestep latents enable temporal anomaly localization, lead-wise attention respects ECG lead geometry, and deep architecture captures hierarchical patterns.

**Limitations**: High parameter count increases memory requirements and training time, bidirectional processing prevents true online inference, multiple attention mechanisms add computational overhead, and risk of overfitting on smaller datasets requires strong regularization.

### 8.4 Variational Autoencoder with Bidirectional LSTM and Multi-Head Attention (VAE-BiLSTM-MHA)

This model represents a refinement of the VAE-BiLSTM-Attention architecture with a more streamlined attention mechanism and improved training stability through architectural enhancements.

**Architecture Philosophy**: While similar to VAE-BiLSTM-Attention, this model emphasizes cleaner separation between encoding, attention, and decoding stages. The architecture implements a three-stage pipeline where encoding extracts features, attention refines them, and decoding reconstructs signals.

**Encoder Architecture**: The encoder uses a two-layer bidirectional LSTM with hidden dimensions five hundred twelve and two hundred fifty-six. Gaussian noise with minimal standard deviation (zero point zero zero one) is added to inputs for regularization. The encoder produces per-timestep latent distributions with means and log-variances. Layer normalization and dropout at zero point one improve training stability.

**Multi-Head Attention**: The attention mechanism projects the input ECG to query vectors using a learned linear transformation, then applies multi-head attention with eight heads. Unlike the previous model, this uses a single unified attention mechanism rather than separate temporal and lead-wise attention. The attention operates on query-key-value triplets where queries come from input projection and values come from the latent codes. This design allows the model to learn what aspects of the input should attend to which parts of the latent representation.

**Decoder Architecture**: The decoder architecture mirrors the encoder with two bidirectional LSTM layers (dimensions two hundred fifty-six and five hundred twelve). It processes the attention-refined latent codes to generate reconstructions. Output layers include layer normalization, PReLU activations, and dropout for stability. The final layer produces mean and log-variance for each timestep and lead.

**Loss Function**: Uses mean squared error for reconstruction and KL divergence for regularization with beta of zero point zero one. The per-timestep KL divergence computation allows anomaly localization while the global aggregation ensures stable training.

**Key Characteristics**: Contains approximately ten million parameters similar to VAE-BiLSTM-Attention. Implements pre-layer normalization for better gradient flow. Uses PReLU activations which learn optimal activation shapes. Includes cyclical beta scheduling that varies beta across training to prevent collapse.

**Strengths**: Cleaner architecture with well-separated stages, improved training stability through layer normalization, unified attention mechanism is simpler than dual attention, PReLU activations adapt to data characteristics, and cyclical beta scheduling prevents KL collapse.

**Limitations**: Still computationally expensive with ten million parameters, single attention mechanism may be less powerful than dual attention, requires careful beta scheduling hyperparameter tuning, and bidirectional processing prevents true streaming inference.

### 8.5 Beat-Aligned Variational Autoencoder (BA-VAE)

The Beat-Aligned VAE introduces a fundamentally different approach by operating on individual heartbeat windows rather than full ECG segments. This beat-centric design aligns with clinical ECG interpretation practices where cardiologists analyze individual cardiac cycles.

**Architecture Philosophy**: Rather than processing arbitrary windows of ECG data, BA-VAE segments signals into beat-aligned windows centered on R-peaks (the most prominent feature in ECG corresponding to ventricular depolarization). This alignment ensures each input represents a complete cardiac cycle, making abnormality detection more interpretable and aligned with clinical practice.

**Beat Extraction**: The preprocessing pipeline identifies R-peaks in Lead II using scipy's peak finding algorithm with minimum distance constraints of three hundred milliseconds. Around each R-peak, a window is extracted spanning two hundred milliseconds before and four hundred milliseconds after the R-peak. If the resulting window is shorter or longer than the target length (one thousand samples to match other models), it is padded or center-cropped accordingly. This process converts variable-length ECG recordings into a dataset of fixed-size beat windows.

**Encoder Architecture**: The encoder processes beat windows using a two-layer bidirectional GRU with hidden dimension two hundred fifty-six. The bidirectional design captures both the rising and falling phases of cardiac waveforms. Layer normalization after the GRU improves training stability. The final hidden state is processed through a fully connected network with GELU activations and dropout to produce mean and log-variance for a thirty-two dimensional latent space. The architecture is deliberately scaled down compared to BiLSTM models since beat-level patterns are less complex than multi-beat sequences.

**Decoder Architecture**: The decoder expands the sampled latent code across all timesteps in the beat window, creating a repeated latent input sequence. A unidirectional GRU with hidden dimension two hundred fifty-six processes this sequence with an initial hidden state derived from the latent code through a learned transformation. Layer normalization stabilizes the recurrent processing. The output sequence is transformed through a fully connected network to produce mean and log-variance reconstructions for all twelve leads at each timestep.

**Loss Function**: Uses mean squared error for reconstruction loss rather than Gaussian negative log-likelihood to prevent issues with learned high variance predictions. KL divergence includes a free-bits constraint (zero point zero one per dimension) to prevent individual latent dimensions from collapsing to the prior. Beta weighting of zero point three provides stronger regularization than the larger models.

**Key Characteristics**: Contains approximately five hundred thirty-four thousand parameters, making it more lightweight than BiLSTM models. Operates on beat-aligned windows which provides clinical interpretability. The beat extraction process enables analysis of beat-to-beat variability. Can process individual beats in isolation for fast inference.

**Strengths**: Clinically interpretable beat-level analysis, lightweight architecture enables fast training and inference, beat alignment improves feature consistency, can analyze beat-to-beat variability patterns, and beat-level focus simplifies anomaly attribution.

**Limitations**: Requires accurate R-peak detection which may fail on heavily corrupted signals, loses inter-beat timing information which is clinically relevant, beat extraction preprocessing adds computational overhead, may miss arrhythmias that manifest across multiple beats, and shorter sequence length provides less context.

### 8.6 Hierarchical Latent Variational Autoencoder (HL-VAE)

The Hierarchical Latent VAE implements a multi-scale latent representation with separate global and local latent variables, enabling the model to capture both overall signal characteristics and fine-grained local patterns simultaneously.

**Architecture Philosophy**: Clinical ECG interpretation naturally occurs at multiple scales. Global characteristics include heart rate, rhythm regularity, and overall morphology, while local features include specific wave shapes, intervals, and segments. HL-VAE explicitly models this hierarchy through dual latent spaces that capture global (sequence-level) and local (timestep-level) representations.

**Global Encoder**: The global encoder uses a two-layer bidirectional LSTM with hidden dimensions five hundred twelve and two hundred fifty-six to process the entire sequence. The final hidden states from both directions are averaged across time to produce a global context vector representing the entire ECG segment. This vector is transformed through a fully connected network with GELU activations and dropout to produce mean and log-variance for a sixty-four dimensional global latent space.

**Local Encoder**: Operating in parallel, the local encoder uses a single-layer bidirectional LSTM with hidden dimension one hundred twenty-eight to generate per-timestep features. Each timestep's hidden state is independently transformed through a small fully connected network to produce mean and log-variance for a sixteen dimensional local latent space. This creates a sequence of local latent codes, one per timestep, capturing fine-grained temporal variations.

**Decoder Architecture**: The decoder receives both global and local latent codes. The global latent is expanded across all timesteps and concatenated with the per-timestep local latents, creating a combined representation with dimensions eighty (sixty-four global plus sixteen local). This hierarchical code is processed through a two-layer bidirectional LSTM decoder with hidden dimensions two hundred fifty-six and five hundred twelve. The output layer generates reconstruction mean and log-variance for all timesteps and leads.

**Loss Function**: The loss function includes three components. Reconstruction loss uses mean squared error between predicted and actual ECG values. Global KL divergence regularizes the global latent space with beta weight of zero point zero one. Local KL divergence regularizes the per-timestep local latents with a separate beta weight of zero point zero one. This dual regularization encourages the model to use both latent spaces meaningfully rather than collapsing one while over-using the other.

**Key Characteristics**: Contains approximately five million parameters with the dual-encoder architecture. The hierarchical latent structure enables interpretable anomaly attribution (is the issue global rhythm or local morphology?). Training requires balancing two beta parameters to ensure both latent spaces are utilized. The model can generate novel ECGs by manipulating global and local codes independently.

**Strengths**: Multi-scale latent representation matches clinical interpretation, can localize anomalies to global versus local characteristics, enables fine-grained control through hierarchical sampling, dual encoding captures complementary information, and the architecture supports interpretable anomaly analysis.

**Limitations**: Increased architectural complexity requires careful hyperparameter tuning, dual beta parameters add optimization challenges, higher parameter count than simpler models, training requires ensuring both latent spaces remain active, and inference is slower due to dual encoding paths.

### 8.7 Spectral-Temporal Variational Autoencoder (ST-VAE)

ST-VAE represents one of the most sophisticated architectures in this study, implementing dual-domain learning that processes ECG signals in both time and frequency domains simultaneously. This model achieves state-of-the-art performance through its innovative multi-domain approach.

**Architecture Philosophy**: ECG signals contain information at multiple temporal scales and frequencies. Morphological abnormalities (such as myocardial infarction) manifest as shape changes in the time domain, while rhythmic abnormalities (such as arrhythmias) appear as altered frequency patterns. By processing both domains, ST-VAE captures complementary information that single-domain models miss.

**Time Domain Encoder**: The time encoder implements a ResNet-style architecture with four residual blocks. Each block contains two convolutional layers with batch normalization and GELU activations, plus a skip connection for gradient flow. Channel dimensions progress from input (twelve leads) through thirty-two, sixty-four, one hundred twenty-eight, to two hundred fifty-six feature maps. Strided convolutions with kernel sizes seven, five, five, and three provide progressive downsampling. The final feature maps are adaptively pooled to a fixed size and flattened, producing an eight thousand one hundred ninety-two dimensional time feature vector. Dropout at zero point two provides regularization.

**Frequency Domain Encoder**: The frequency encoder first computes the real Fast Fourier Transform (FFT) of the input signal along the temporal dimension. For a one thousand sample input, this produces five hundred one frequency bins per lead. The FFT magnitude is normalized per-lead by dividing by the maximum value, then log-transformed using log1p for dynamic range compression. The normalized log-magnitude spectrum is processed through three ResNet blocks with channels thirty-two, sixty-four, and one hundred twenty-eight. Adaptive pooling and flattening produce a four thousand ninety-six dimensional frequency feature vector.

**Fusion and Latent Projection**: The time and frequency features are concatenated into a twelve thousand two hundred eighty-eight dimensional combined representation. This fused vector is processed through a fusion network consisting of a fully connected layer with five hundred twelve outputs, layer normalization, GELU activation, and dropout. The fusion network output is split into mean and log-variance for a thirty-two dimensional latent space. The relatively small latent dimension compared to the large input representation forces the model to learn compressed, meaningful features.

**Decoder Architecture**: The decoder operates purely in the time domain. The sampled latent code is transformed through a fully connected layer to produce an initial hidden representation with dimensions two hundred fifty-six by sequence-length-divided-by-eight. This is reshaped into a spatial feature map and processed through four transpose convolutional layers with channels two hundred fifty-six, one hundred twenty-eight, sixty-four, and thirty-two. Each layer uses kernel size four with stride two for upsampling. The final convolution projects to twelve leads at the original one thousand sample length. If the output size doesn't exactly match, bilinear interpolation adjusts it to one thousand samples.

**Multi-Domain Loss Function**: The loss function uniquely combines multiple reconstruction objectives. Time domain loss computes mean squared error between reconstructed and original ECG in the time domain. Frequency domain loss computes L1 distance between the log-normalized FFT magnitudes of the reconstruction and original signal. L1 is used for frequency loss as it is more robust to outliers than MSE. Spectral regularization adds an additional MSE loss on low-frequency components (zero to fifty Hertz) which contain most clinically relevant information. The frequency loss weight is a learnable parameter initialized to zero point three but allowed to adapt during training through a sigmoid-bounded parameter. KL divergence with free bits regularization (zero point zero one per dimension) prevents latent collapse. The beta parameter of zero point five provides strong regularization.

**Key Characteristics**: Contains approximately eight million parameters with the dual ResNet encoders. Training uses cyclical beta annealing with ten-epoch cycles ramping from zero point one to zero point five. The learnable frequency weight adapts the time-frequency balance based on dataset characteristics. The model achieved the best performance in this study with F1 score of zero point six five seven and AUC-ROC of zero point eight four eight.

**Strengths**: Dual-domain processing captures complementary information, ResNet backbone provides excellent gradient flow, frequency domain captures global rhythm patterns, time domain captures local morphological features, learnable frequency weighting enables automatic balancing, strong performance with robust generalization, and multi-domain loss prevents mode collapse.

**Limitations**: Large parameter count requires more training data and compute, FFT computation adds marginal overhead, dual encoders increase model complexity, training requires balancing three loss components, and the complex architecture requires careful hyperparameter tuning.

### 8.8 Diffusion Variational Autoencoder (Diffusion-VAE)

The Diffusion-VAE incorporates concepts from diffusion probabilistic models into the VAE framework, implementing a forward diffusion process that gradually adds noise and a learned reverse denoising process. This approach represents cutting-edge research in generative modeling applied to ECG analysis.

**Architecture Philosophy**: Diffusion models have achieved remarkable success in image generation by learning to reverse a gradual noising process. Diffusion-VAE applies this paradigm to ECG signals, hypothesizing that learning to denoise corrupted ECGs will produce robust representations useful for anomaly detection. The denoising task forces the model to learn what constitutes valid ECG structure versus noise or abnormalities.

**Forward Diffusion Process**: During training, clean ECG signals are corrupted through a Markov chain that gradually adds Gaussian noise over one hundred timesteps. The noise schedule follows a linear schedule where beta parameters increase from zero point zero zero zero one to zero point zero two. At each diffusion timestep t, the noisy signal x_t is computed as a linear combination of the original signal x_0 and Gaussian noise, weighted by schedule parameters alpha_t. This process converts any ECG into pure Gaussian noise by the final timestep.

**VAE Encoder**: The base encoder uses a two-layer bidirectional GRU with hidden dimension two hundred fifty-six to process the input ECG. The final hidden state is transformed through a fully connected network with layer normalization, GELU activations, and dropout to produce mean and log-variance for a thirty-two dimensional latent space. This provides a compressed representation of the original clean signal.

**VAE Decoder**: The decoder receives sampled latent codes and generates reconstructions through a symmetric GRU-based architecture. The latent code is first projected to an initial hidden state through a fully connected network with layer normalization. This hidden state initializes a two-layer GRU with hidden dimension two hundred fifty-six. The latent code is expanded across all timesteps as input to the GRU. The GRU output sequence is transformed through a final fully connected layer to produce reconstruction mean and log-variance.

**Denoising Network**: The denoising network implements a simplified U-Net style architecture. It takes a noisy ECG sample x_t and diffusion timestep t as input. The timestep is embedded using sinusoidal positional encoding (same as in transformers), then processed through a multi-layer perceptron. The ECG is encoded through three convolutional layers with SiLU activations. The timestep embedding is added to the encoded features through broadcast addition. A symmetric decoder with three transpose convolutions reconstructs the predicted noise. The network is trained to predict the Gaussian noise component that was added to create x_t from x_0.

**Combined Loss Function**: The total loss combines three components. VAE reconstruction loss uses mean squared error between the decoder output and original ECG. KL divergence regularizes the latent space with beta weight zero point three. Diffusion denoising loss uses mean squared error between the predicted noise and the actual noise added during forward diffusion, weighted by zero point five. During training, random diffusion timesteps are sampled for each batch, and the model learns to denoise across all noise levels.

**Key Characteristics**: Contains approximately two million nine hundred thousand parameters split between the VAE and denoising networks. Training alternates between VAE objectives and diffusion objectives. The one hundred-step diffusion process during inference can be shortened for faster sampling. The dual training objectives provide complementary learning signals.

**Strengths**: Denoising pretraining creates robust features, learns what constitutes valid ECG structure, can generate novel samples through diffusion sampling, combines benefits of VAEs and diffusion models, and the multi-task learning improves generalization.

**Limitations**: Training complexity with dual objectives, longer inference time due to diffusion sampling steps, requires careful balancing of VAE and diffusion loss weights, one hundred diffusion steps add memory overhead, and the architecture requires more training epochs to converge.

### 8.9 Transformer Variational Autoencoder (Transformer-VAE)

The Transformer-VAE applies the transformer architecture, revolutionary in natural language processing, to ECG analysis. This model replaces recurrent networks with self-attention mechanisms for parallel processing and explicit long-range dependency modeling.

**Architecture Philosophy**: Transformers process entire sequences in parallel through self-attention, avoiding the sequential bottleneck of RNNs. For ECG analysis, this means the model can simultaneously attend to distant features like the relationship between P-waves and T-waves without propagating information through intermediate timesteps. The multi-head attention mechanism enables learning multiple complementary representations.

**Input Processing**: Raw ECG input with shape (batch, one thousand timesteps, twelve leads) is first projected to the model dimension through a learned linear transformation. The twelve lead values at each timestep are mapped to a two hundred fifty-six dimensional embedding. Sinusoidal positional encodings are added to these embeddings to inject temporal order information, since attention operations are permutation-invariant without positional information.

**Encoder Architecture**: The transformer encoder consists of four encoder layers. Each layer implements multi-head self-attention with eight heads followed by a position-wise feed-forward network. The attention mechanism computes scaled dot-product attention where each position attends to all positions in the input. The eight heads learn different attention patterns in parallel (such as attending to previous beats, following morphological changes, or tracking rhythm variations). The feed-forward network has dimension one thousand twenty-four (four times the model dimension, following standard transformer ratios) with GELU activation. Pre-layer normalization (layer norm before attention and FFN rather than after) improves training stability. Residual connections around both attention and FFN enable gradient flow.

**Latent Space Projection**: The encoder output sequence is flattened from (batch, one thousand, two hundred fifty-six) to (batch, two hundred fifty-six thousand). This flat representation is processed through a temporal pooling network that reduces it to a two hundred fifty-six dimensional global context vector. This vector is projected to mean and log-variance for a thirty-two dimensional latent space. The compression from two hundred fifty-six thousand dimensions to thirty-two forces the model to extract only the most salient features.

**Decoder Architecture**: The decoder expands the sampled latent code back to a sequence. The latent is first projected through a fully connected network to produce a flattened sequence representation of dimension two hundred fifty-six thousand. This is reshaped to (batch, one thousand, two hundred fifty-six). Sinusoidal positional encodings are added, and the sequence is processed through four decoder layers. Each decoder layer has the same structure as encoder layers: multi-head self-attention with eight heads followed by position-wise FFN with dimension one thousand twenty-four. Pre-layer normalization and residual connections are used throughout.

**Output Projection**: The decoder output sequence is projected through a final network to produce reconstruction mean and log-variance. This network includes a fully connected layer with two hundred fifty-six outputs, layer normalization, GELU activation, dropout at zero point one, and a final fully connected layer producing twenty-four outputs (two times twelve leads for mean and log-variance). The output has shape (batch, one thousand, twelve) matching the input ECG.

**Loss Function**: Uses mean squared error for reconstruction loss and KL divergence with beta weight zero point three for latent regularization. The loss is standard VAE loss without the multi-domain components of ST-VAE.

**Key Characteristics**: Contains approximately one hundred thirty-seven million parameters, making it the largest model by far. The transformer architecture enables true parallel processing without sequential dependencies. Training benefits from modern techniques like pre-layer normalization and careful initialization. The model's capacity is very high but requires large datasets and regularization to avoid overfitting.

**Strengths**: Parallel processing enables fast training on GPUs, self-attention explicitly models long-range dependencies, multi-head attention learns diverse representations, no recurrent bottleneck for long sequences, highly flexible architecture proven across domains, and positional encoding preserves temporal information.

**Limitations**: Extremely large parameter count (one hundred thirty-seven million) requires massive datasets, quadratic complexity of attention (O(n²) in sequence length) limits scalability, high memory requirements during training, risk of overfitting on smaller medical datasets, requires extensive hyperparameter tuning, and large model size prevents edge deployment.

### 8.10 Model Comparison Summary

**By Architecture Family**:
- **Convolutional**: CAE (baseline, lightweight, fast)
- **Recurrent**: VAE-GRU, VAE-BiLSTM-Attention, VAE-BiLSTM-MHA (temporal modeling)
- **Hybrid**: BA-VAE (beat-centric), HL-VAE (hierarchical), ST-VAE (multi-domain)
- **Advanced**: Diffusion-VAE (denoising), Transformer-VAE (attention-based)

**By Parameter Count**:
- Small (<1M): CAE (250K), BA-VAE (534K)
- Medium (1-10M): VAE-GRU, VAE-BiLSTM models (2-10M), ST-VAE (8M)
- Large (>10M): Transformer-VAE (137M)

**By Computational Cost**:
- Fastest: CAE, BA-VAE (simple architectures)
- Medium: VAE-GRU, ST-VAE (moderate complexity)
- Slowest: BiLSTM models (bidirectional), Transformer-VAE (attention), Diffusion-VAE (sampling)

**By Performance** (based on available results):
- **Best**: ST-VAE (F1=0.657, AUC=0.848) - multi-domain learning excels
- **Good**: BA-VAE (F1=0.554, AUC=0.778) - beat alignment helps
- **Moderate**: BiLSTM models (F1≈0.52, AUC≈0.78) - strong but computationally expensive
- **Baseline**: CAE, Diffusion-VAE (work in progress)

**By Clinical Interpretability**:
- **High**: BA-VAE (beat-level), ST-VAE (time-frequency separation)
- **Medium**: HL-VAE (hierarchical), Attention models (attention weights)
- **Low**: CAE (black box), standard VAEs (latent space less interpretable)

All models follow a common training interface enabling fair comparison. Each architecture offers unique trade-offs between accuracy, speed, interpretability, and computational requirements, making them suitable for different deployment scenarios from real-time edge devices to cloud-based research platforms.
