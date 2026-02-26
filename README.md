# Multi-bit Black-Box Watermarking of Deep Neural Networks

Course project for **Machine Learning Security** at Sapienza University of Rome, MSc Cybersecurity.

This project implements the multi-bit watermarking scheme proposed by:

> **S. Leroux, S. Vanassche, and P. Simoens**, *"Multi-bit, Black-box Watermarking of Deep Neural Networks in Embedded Applications"*, CVPR Workshop on Responsible Generative AI, 2024.

The implementation also draws on foundational concepts from:

> **Y. Adi, C. Baum, M. Cissé, B. Pinkas, and J. Keshet**, *"Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring"*, USENIX Security Symposium, 2018.

---

## Overview

Neural network watermarking allows a model owner to embed a hidden signature into a trained model. This signature can later be extracted to prove ownership or trace the source of a leaked model, without needing access to the model's internal weights (black-box verification).

The Leroux et al. scheme improves upon prior work (Adi et al.) by enabling **multi-bit** watermarking: from a single base model and a single set of trigger inputs, it is possible to generate a large number of uniquely watermarked model instances. Each instance encodes a different binary signature, making it possible to identify *which specific copy* was leaked and not just that a leak occurred. This is particularly relevant for edge AI deployment, where distinct model copies are distributed to individual users or devices.

## Method

The watermarking pipeline consists of three phases, following the structure of the Leroux et al. paper:

### Phase 1 - Base Model Training

A ResNet-18 classifier is trained on CIFAR-10 to serve as the base (unwatermarked) model. The architecture is adapted for 32×32 images by replacing the first 7×7 convolution with a 3×3 convolution and removing the initial max pooling layer. The Leroux paper uses a Wide Residual Network, but the watermarking scheme is architecture-agnostic.

### Phase 2 - Key Generation

This phase produces the cryptographic "key" material needed for watermarking:

**Encoding scheme (Section 3.1 of the paper):** The model's output logits are collected for a subset of training samples, and the mean logit vector is computed for each class. K-means clustering (K=2) splits the 10 CIFAR-10 classes into two groups, each mapped to a bit value (0 or 1). In practice, this naturally separates animals (bit 0) from vehicles (bit 1), reflecting the model's internal representation structure.

**Trigger input generation (Section 3.2, Equation 1):** 30 trigger inputs are generated, each optimized to lie on the decision boundary between one class from the bit-0 group and one from the bit-1 group. The optimization minimizes a loss function that simultaneously pushes two target class logits above a threshold while suppressing all other class logits. The result is an input that the model is equally uncertain about, sitting exactly on the decision boundary.

### Phase 3 - Signature Embedding and Extraction

**Embedding (Section 3.3):** To embed a specific 30-bit watermark, a copy of the base model is fine-tuned on the original training data combined with the trigger inputs, where each trigger is labeled according to the desired watermark bit. Since the triggers already sit on the decision boundary, only minimal weight adjustments are needed to push them to the correct side.

**Extraction (Section 3.4):** To extract a watermark, the trigger inputs are fed to the suspected model and the predicted classes are mapped through the encoding scheme to recover the bitstring. This requires only query access (black-box).

### Robustness Evaluation

Four attacks are evaluated to test the watermark's resilience:

- **Fine-Tuning (FT):** Fine-tune all model weights on the original training set.
- **Retrain Last Layer (RT):** Reinitialize and retrain the last fully connected layer, then fine-tune all weights.
- **Fine-Pruning (FP):** Zero out the smallest weights by magnitude, then fine-tune to recover accuracy.
- **Input Noising (IN):** Add Gaussian noise to inputs at inference time.

## Results

### Watermark Embedding

| Model | Test Accuracy | Watermark Accuracy |
|-------|:---:|:---:|
| Base (no watermark) | 93.53% | N/A |
| Instance A | 93.43% | 100% |
| Instance B | 93.57% | 100% |
| Instance C | 93.31% | 100% |

Watermarking causes negligible accuracy degradation (≤0.22%). All 30 bits are correctly embedded and extracted from each instance, and all three instances produce distinct watermarks when queried with the same trigger inputs. With K=30 trigger inputs, the scheme supports up to 2^30 ≈ 1 billion unique model instances.

### Robustness

| Attack | Model Accuracy | Watermark Accuracy |
|--------|:---:|:---:|
| Before attacks | 93.43% | 100.0% |
| Fine-Tune (15 epochs) | 93.4% | 100.0% |
| Retrain Last Layer (15 epochs) | 92.1% | 66.7% |
| Fine-Pruning (90%) | 92.4% | 76.7% |
| Input Noising (σ=0.10) | 89.8% | 100.0% |

The watermark is fully resilient to fine-tuning and input noising attacks (100% accuracy preserved). The Retrain Last Layer attack is the most effective, reducing watermark accuracy to 66.7% (10 bits flipped out of 30), but this is still well above the 50% random baseline and comes at the cost of reduced model accuracy. Fine-pruning at an aggressive 90% level degrades the watermark to 76.7% while the model retains 92.4% accuracy. In both partially successful attacks, the attacker faces a fundamental trade-off: removing the watermark requires modifications that also damage the model's primary functionality.

## How to Run

### Prerequisites

- Python 3.8 or higher
- (Optional) NVIDIA GPU with CUDA support for faster training

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Create a virtual environment
python -m venv watermarking-env
source watermarking-env/bin/activate  # Linux/Mac
# watermarking-env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

If you have a CUDA-compatible GPU, install PyTorch with GPU support first by following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/), then run `pip install -r requirements.txt` to install the remaining packages.

### Run the notebook

```bash
jupyter lab multibit_watermarking_project.ipynb
```

Run all cells sequentially. The CIFAR-10 dataset (~170 MB) will be downloaded automatically on first run.

## Project Structure

```
.
├── multibit_watermarking_project.ipynb   # Main notebook
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
└── README.md                              # This file
```

Files generated at runtime (not tracked in git):
- `data/` - CIFAR-10 dataset (auto-downloaded)
- `base_model.pth` - Trained base model checkpoint

## Acknowledgments

Aim of this project was primarily the understanding of the theoretical concepts of the topic, and implementation was auxiliary to better understand the concepts explained in the paper. Because of this, the project was developed with the assistance of **Claude Opus 4.6** (Anthropic). Claude helped with the project design, code implementation, debugging, and analysis of results.

## References

1. S. Leroux, S. Vanassche, and P. Simoens, "Multi-bit, Black-box Watermarking of Deep Neural Networks in Embedded Applications", *CVPR Workshop on Responsible Generative AI*, 2024.
2. Y. Adi, C. Baum, M. Cissé, B. Pinkas, and J. Keshet, "Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring", *27th USENIX Security Symposium*, 2018.
