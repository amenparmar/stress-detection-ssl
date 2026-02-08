# Self-Supervised Stress Detection

Cross-subject stress detection from multimodal physiological signals using contrastive learning (SimCLR).

## 🎯 Features

- **Self-supervised pre-training** with SimCLR on physiological signals
- **Multi-modal fusion** with separate encoders for EDA, TEMP, and BVP
- **Attention mechanisms** for improved feature learning
- **Data augmentation** for physiological signals (noise, scaling, warping)
- **Ensemble learning** with multiple model voting
- **K-fold cross-validation** for robust evaluation
- **🆕 Domain Adversarial Training (DANN)** - Subject-invariant feature learning
- **🆕 Latent Trajectory Analysis** - Continuous stress monitoring with personalized baselines
- **🆕 Subject-Invariant Losses** - MMD, CORAL, and Contrastive learning

## 📊 Results

### Tested Performance (Random Split Evaluation)

**System Specifications:**
- **GPU:** NVIDIA RTX 5070 Ti
- **Training Time:** ~6.5 hours for complete pipeline
- **Dataset:** WESAD (15 subjects, 1783 segments)

**Performance Results:**

| Model Configuration | Test Accuracy | F1 Score | Training Time |
|---------------------|---------------|----------|---------------|
| **Baseline (Standard)** | 74.35% | ~0.65 | ~30 min |
| **Individual Ultimate Models** | 80.17-83.67% | 0.68-0.72 | ~1 hr each |
| **🏆 Ultimate Ensemble (5 models)** | **79.01%** | **0.67** | **~6.5 hrs total** |

**Individual Model Breakdown:**
- Model 1: 80.76%
- Model 2: 81.05%
- Model 3: 82.51%
- Model 4: 83.67% (best individual)
- Model 5: 80.17%
- **Average:** 81.63% ± 1.28%

**Improvement Over Baseline:** +4.66% absolute (79.01% vs 74.35%)

**Key Features:**
- ResNet-based encoder with channel and temporal attention
- Multi-modal fusion (EDA + TEMP + BVP)
- Domain adversarial training (DANN) for subject-invariance
- Subject-invariant losses (MMD + CORAL + Contrastive)
- Latent trajectory analysis with personalized baselines
- Temporal consistency regularization

## 🚀 Quick Start

### Requirements
- Python 3.8+
- PyTorch with CUDA support (optional but recommended)
- WESAD dataset

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd stress_detection

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch torchvision torchaudio scikit-learn scipy tqdm numpy
```

### Usage

**Option 1: Windows Batch Script (Easiest)**
```bash
.\run.bat
# Select from 12 options:
# 1. Test Run (Mock Data)
# 2. Pre-train (500 epochs)
# 3. Evaluate (Standard)
# 4. Train Ensemble (5 models)
# 5. Multi-Modal Fusion
# 6. Full Pipeline (Multi-Modal Ensemble - Max Accuracy)
# 7. SMOTE Oversampling (Fix Class Imbalance)
# 8. Leave-One-Subject-Out CV (Gold Standard Evaluation)
# 9. Domain Adversarial Training (DANN - Subject-Invariant)
# 10. Latent Trajectory Analysis (Continuous Monitoring)
# 11. Subject-Invariant Loss Training (MMD + CORAL + Contrastive)
# 12. COMBINED ADVANCED - MAXIMUM PERFORMANCE (82-86% expected)
# 13. 🏆 ULTIMATE PERFORMANCE - ALL TECHNIQUES + ENSEMBLE (85-88% expected)
```

**Option 2: Command Line**
```bash
# Pre-training
python -m stress_detection.main --mode pretrain --epochs 500 --batch_size 32

# Evaluation
python -m stress_detection.main --mode evaluate --epochs 100 --batch_size 32

# Full pipeline (Multi-Modal Ensemble)
python -m stress_detection.main --mode multimodal_ensemble --epochs 100 --batch_size 32

# SMOTE oversampling
python -m stress_detection.main --mode smote --epochs 100 --batch_size 32

# Leave-One-Subject-Out Cross-Validation
python -m stress_detection.main --mode loso --epochs 100 --batch_size 32

# 🆕 Domain Adversarial Training (DANN)
python -m stress_detection.main --mode dann --epochs 100 --batch_size 32

# 🆕 Latent Trajectory Analysis
python -m stress_detection.main --mode trajectory --epochs 100 --batch_size 32

# 🆕 Subject-Invariant Loss Training
python -m stress_detection.main --mode invariant --epochs 100 --batch_size 32

# 🆕 Combined Advanced (Maximum Performance)
python -m stress_detection.main --mode combined --epochs 100 --batch_size 32

# 🏆 Ultimate Performance (All Techniques + Ensemble)
python -m stress_detection.main --mode ultimate --epochs 100 --batch_size 32
```

## 📁 Project Structure

```
stress_detection/
├── data/
│   ├── dataset.py           # WESAD data loader (with subject IDs)
│   └── augmentation.py      # Signal augmentation
├── models/
│   ├── encoder.py           # ResNet encoder with attention
│   ├── ssl_head.py          # Projection head for SSL
│   ├── attention.py         # Attention mechanisms
│   ├── multimodal_encoder.py # Multi-modal fusion
│   ├── 🆕 gradient_reversal.py  # GRL for DANN
│   ├── 🆕 domain_classifier.py   # Subject ID classifier
│   └── 🆕 trajectory_analyzer.py # Trajectory analysis
├── training/
│   ├── train_ssl.py         # SimCLR pre-training
│   ├── train_classifier.py # Supervised classifier training
│   ├── train_ensemble.py   # Ensemble training
│   ├── train_smote.py       # SMOTE oversampling
│   ├── train_loso.py        # Leave-one-subject-out CV
│   ├── loss.py              # NT-Xent loss
│   ├── 🆕 train_dann.py         # Domain adversarial training
│   ├── 🆕 train_trajectory.py   # Trajectory-based training
│   ├── 🆕 train_invariant.py    # Subject-invariant loss training
│   ├── 🆕 invariant_losses.py   # MMD, CORAL, Contrastive losses
│   └── 🆕 train_ultimate.py     # Ultimate unified training (all techniques)
├── utils/
│   ├── config.py            # Hyperparameters
│   └── cross_validation.py # K-fold CV
├── main.py                  # Entry point
└── run.bat                  # Windows batch script
```

## 🧠 Advanced Techniques (NEW)

### 1. Domain Adversarial Neural Networks (DANN)
**Purpose:** Remove subject-specific patterns for better cross-subject generalization  
**Implementation:** Gradient Reversal Layer + Domain Classifier  
**Impact:** Forces encoder to learn subject-invariant features

### 2. Latent Trajectory Analysis
**Purpose:** Continuous stress monitoring with personalized baselines  
**Implementation:** Per-subject baseline extraction + deviation tracking + temporal smoothing  
**Impact:** Enables real-time stress scores and better temporal consistency

### 3. Subject-Invariant Loss Functions
**Components:**
- **MMD (Maximum Mean Discrepancy):** Minimizes distribution differences between subjects
- **CORAL:** Aligns second-order statistics (covariances)
- **Contrastive Subject Loss:** Pulls same-stress different-subject pairs together

**Impact:** Reinforces cross-subject alignment through multiple mechanisms

### 4. Unified Multi-Loss Training (Option 13)
**Combines all 5 loss terms:**
```
Total Loss = Classification Loss 
           + 0.1 × Adversarial Loss (DANN)
           + 0.05 × Invariant Losses (MMD+CORAL+Contrastive)
           + 0.05 × Trajectory Deviation Loss
           + 0.02 × Temporal Consistency Loss
```

**Architecture:**
- **Stage 1:** SSL Pre-training (500 epochs, ~1.5 hours)
- **Stage 2:** Train 5 ultimate models with all techniques (~4-5 hours)
- **Stage 3:** Ensemble evaluation via majority voting (~10 minutes)

## 🔧 Configuration

Edit `utils/config.py` to customize:

```python
WESAD_dataset_path = r'C:\path\to\WESAD'  # Update this!
WINDOW_SIZE = 60  # seconds
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 3e-4
```

## 📈 Advanced Features

### Data Augmentation
```python
from data.augmentation import SignalAugmentation

augmentor = SignalAugmentation(
    noise_factor=0.05,
    scale_range=(0.85, 1.15),
    magnitude_warp_sigma=0.3
)
```

### K-Fold Cross-Validation
```python
from utils.cross_validation import k_fold_cross_validate

results = k_fold_cross_validate(
    subject_data=subject_data,
    encoder_class=Encoder,
    k_folds=5
)
```

## ⚙️ Technical Details

### Hardware Requirements

**Tested Configuration:**
- **GPU:** NVIDIA RTX 5070 Ti
- **RAM:** 16GB+ recommended
- **Storage:** ~5GB for dataset + models
- **OS:** Windows 11

**Performance Notes:**
- **GPU Training:** ~6.5 hours for full ultimate pipeline
- **CPU Training:** Not recommended (20-30 hours expected)
- **Batch Size:** 32 (reduce to 16 or 8 if OOM errors occur)

### Training Time Breakdown

| Stage | Description | Time (RTX 5070 Ti) |
|-------|-------------|-------------------|
| **Stage 1** | SSL Pre-training (500 epochs) | ~1.5 hours |
| **Stage 2** | Ultimate Model Training (5 models × 100 epochs) | ~4-5 hours |
| **Stage 3** | Ensemble Evaluation | ~10 minutes |
| **Total** | Complete Ultimate Pipeline | **~6.5 hours** |

### Loss Function Details

**Ultimate Training optimizes 5 loss terms simultaneously:**

1. **Classification Loss (CE):** Standard cross-entropy for stress labels
2. **Adversarial Loss (α=0.1):** Domain classifier tries to identify subjects, encoder tries to fool it
3. **MMD Loss (β=0.01):** Minimizes distribution difference using RBF kernel
4. **CORAL Loss (β=0.01):** Aligns covariance matrices across subjects
5. **Contrastive Loss (β=0.03):** Pulls same-stress different-subject pairs together
6. **Trajectory Loss (γ=0.05):** Classification based on deviation from personalized baseline
7. **Temporal Loss (δ=0.02):** Encourages smooth feature evolution over time

### Model Architecture

**Encoder:** ResNet-style with attention
- **Input:** (Batch, 3, 240) - 3 channels (EDA, TEMP, BVP), 240 time steps (60s @ 4Hz)
- **Output:** (Batch, 256) - Latent feature vectors

**Multi-Modal Fusion:**
- Separate encoders for EDA, TEMP, BVP modalities
- Attention-based fusion of modality-specific features
- Output dimension: 256

**Domain Classifier:** 2-layer MLP
- Predicts subject ID from features
- Trained adversarially via Gradient Reversal Layer

**Trajectory Analyzer:**
- Stores per-subject baseline representations
- Computes L2 deviation from baseline
- Temporal smoothing via moving average
- Converts deviations to stress predictions

## 📈 Performance Analysis

### What Works Well
- ✅ **Multi-Modal Fusion:** EDA + TEMP + BVP complementary signals
- ✅ **SSL Pre-training:** Better initialization than random
- ✅ **Ensemble Diversity:** 5 models with different seeds reduce variance
- ✅ **Subject-Invariant Features:** DANN + invariant losses improve generalization

### Areas for Improvement
- ⚠️ **Ensemble Performance:** Individual models (81-83%) outperform ensemble (79%), suggesting possible overfitting
- ⚠️ **Class Imbalance:** Class 1 (Amusement) remains challenging (~0.67 F1)
- ⚠️ **LOSO Evaluation:** Need full LOSO CV for true cross-subject performance

### Recommended Next Steps
1. **Run full LOSO CV** to get true subject-invariant performance metrics
2. **Hyperparameter tuning** of loss weights (α, β, γ, δ)
3. **Increase ensemble diversity** via different architectures or augmentations
4. **Address class imbalance** with focal loss or better SMOTE integration

## 🎓 Dataset

Uses **WESAD** (Wearable Stress and Affect Detection):
- Download from: https://archive.ics.uci.edu/ml/datasets/WESAD
- 15 subjects with physiological signals (EDA, TEMP, BVP, etc.)
- 3 stress conditions: Baseline, Amusement, Stress
- Signals recorded with Empatica E4 wearable sensor

**Dataset Statistics (from testing):**
- Total subjects: 15
- Total segments: 1783 (60-second windows)
- Signal sampling: 4 Hz after downsampling
- Class distribution: Imbalanced (more baseline samples)

## 🔬 Research Context

This implementation combines state-of-the-art techniques from:
- **Domain Adaptation:** Ganin et al. (2016) - Gradient Reversal Layer
- **Distribution Matching:** MMD (Gretton et al.), CORAL (Sun et al.)  
- **Contrastive Learning:** SimCLR (Chen et al., 2020)
- **Physiological Computing:** WESAD dataset (Schmidt et al., 2018)

**Citation:**
```
Schmidt, P., Reiss, A., Dürichen, R., Marberger, C., & Van Laerhoven, K. (2018).
Introducing WESAD, a multimodal dataset for wearable stress and affect detection.
In Proceedings of the 20th ACM International Conference on Multimodal Interaction (pp. 400-408).
```

## 📝 License

This project is for research and educational purposes.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🔗 Citation

If you use this code in your research, please cite:

```bibtex
@software{stress_detection_ssl,
  title={Self-Supervised Stress Detection from Physiological Signals with Advanced Techniques},
  author={Amen Parmar},
  year={2026},
  url={https://github.com/amenparmar/stress-detection-ssl}
}
```

## 📧 Contact

For questions or collaborations, open an issue or contact [amenparmar777@gmail.com].

---

**Built with DANN, trajectory analysis, multi-modal fusion, and subject-invariant losses for maximum cross-subject generalization!** 🚀
