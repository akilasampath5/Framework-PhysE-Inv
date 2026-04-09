# Framework-PhysE-Inv: Physics-Encoded Inverse Modeling for Arctic Snow Depth Estimation

PhysE-Inv is a physics-encoded inverse modeling framework that combines LSTM sequence modeling, multi-head temporal attention, and contrastive learning to estimate latent snow-depth-related physical parameters from sparse ERA5 observations.

## Status

**Under review**, 2025. Code will be fully released upon acceptance.

## Dataset

The Arctic sea ice dataset is publicly available at:
**[https://zenodo.org/records/15665532](https://zenodo.org/records/15665532)**

## Repository Structure

```
Framework-PhysE-Inv/
├── README.md
├── requirements.txt
├── src/
│   ├── models.py       # All model architectures
│   └── utils.py        # Data preprocessing and metrics
└── notebooks/
    ├── Test1_LSTM_MultiheadAttention_works_Ablation_Withcontrastive_sample1.ipynb
    ├── Test1_LSTM_MultiheadAttention_works_Ablation_Withcontrastive_sample2.ipynb
    ├── Test1_LSTM_MultiheadAttention_works_Ablation_Withcontrastive_sample3.ipynb
    ├── Test1_LSTM_MultiheadAttention_works_Ablation_Withoutcontrastive-Sample1.ipynb
    ├── Test1_LSTM_MultiheadAttention_works_Ablation_Withoutcontrastive-Sample2.ipynb
    ├── Test1_LSTM_MultiheadAttention_works_Ablation_Withoutcontrastive-Sample3.ipynb
    ├── Test1_BILSTM_works.ipynb
    ├── Test1_ODE_Attention_works.ipynb
    ├── Test1_Resnet50_works.ipynb
    ├── Test1_LSTM_works.ipynb
    └── Figure-submissionKDD-VVI-Type2.ipynb
```

## Models

| Model | Description |
|-------|-------------|
| `LSTMContrastiveWithAttention` | PhysE-Inv main model |
| `BiLSTMContrastive` | Bidirectional LSTM baseline |
| `ResNet1DContrastive` | 1D ResNet baseline |
| `NeuralODEContrastive` | Neural ODE baseline |

## Framework Overview

PhysE-Inv frames snow depth estimation as a **surjective inverse problem**:
- Multiple sub-grid physical configurations can produce identical ERA5 grid-cell observations
- The framework recovers physically constrained latent parameters {w, b, c} grounded in the hydrostatic balance equation
- Contrastive regularization stabilizes parameter estimation under sparse observations

## Installation

```bash
git clone https://github.com/akilasampath5/Framework-PhysE-Inv.git
cd Framework-PhysE-Inv
pip install -r requirements.txt
```

## Quick Start

```python
from src.models import LSTMContrastiveWithAttention, augment_data, contrastive_loss
from src.utils import load_and_preprocess

# Load and preprocess data
data = load_and_preprocess("spatial_avg_data-ERAsinglelevel_2020-24_daily.csv")

# Initialize model
model = LSTMContrastiveWithAttention(
    input_dim=1, hidden_dim=64, num_layers=2,
    output_dim=1, dropout_rate=0.4, num_heads=4
)
```

See notebooks for full training and evaluation examples.

## Keywords

Physics-Informed Machine Learning · Inverse Modeling · Arctic Snow Depth ·
Contrastive Learning · LSTM · Attention · ERA5 · Sea Ice
