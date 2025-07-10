# TopK Sparse Autoencoder for Gemma2-2b

A comprehensive implementation of TopK Sparse Autoencoders (SAE) for training on Gemma2-2b language model activations using the [nnsight](https://github.com/ndif-team/nnsight) library.

## Features

- **TopK Sparse Autoencoder**: Implements TopK sparsity constraint for interpretable feature learning
- **nnsight Integration**: Seamless activation extraction from Gemma2-2b using nnsight
- **Comprehensive Training**: Full training pipeline with logging, checkpointing, and evaluation
- **Flexible Configuration**: YAML-based configuration management with sensible defaults
- **Rich Logging**: TensorBoard and Weights & Biases integration
- **Type Safety**: Full type annotations and comprehensive testing

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- At least 16GB RAM for Gemma2-2b

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd sae
```

2. Install dependencies using uv:
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

Alternatively, use pip:
```bash
pip install -e .
```

## Quick Start

### 1. Quick Test Run

For a quick test with small-scale settings:

```bash
python scripts/train_topk_sae.py --quick-test
```

This will:
- Use a small subset of data (1K samples)
- Train for 2 epochs with limited steps
- Use smaller model dimensions for faster testing

### 2. Full Training

For full-scale training on Gemma2-2b:

```bash
python scripts/train_topk_sae.py --config configs/gemma2_2b.yaml
```

### 3. Custom Configuration

Create your own configuration file:

```yaml
model:
  input_dim: 2304  # Auto-detected from data
  hidden_dim: 9216  # 4x overcomplete
  k: 128  # Top-k sparsity
  tied_weights: true
  normalize_decoder: true

data:
  model_name: "google/gemma-2-2b"
  layer_name: "model.layers.15"  # Layer to extract activations from
  dataset_name: "openwebtext"
  num_samples: 100000

training:
  num_epochs: 5
  learning_rate: 0.001
  batch_size: 256
  use_wandb: true
  wandb_project: "my-topk-sae"
```

Then run:
```bash
python scripts/train_topk_sae.py --config my_config.yaml
```

## Architecture Overview

### TopK Sparse Autoencoder

The TopK SAE enforces sparsity by keeping only the top-k activations in the hidden layer:

```
Input (d_model) → Encoder → Hidden (d_hidden) → TopK → Sparse Hidden → Decoder → Reconstruction (d_model)
```

Key features:
- **Overcomplete representation**: `d_hidden > d_model` for better feature learning
- **TopK sparsity**: Exactly `k` neurons active per sample
- **Tied weights**: Optional weight tying between encoder and decoder
- **Weight normalization**: Decoder weights normalized to unit norm

### Training Pipeline

1. **Activation Collection**: Extract activations from Gemma2-2b using nnsight
2. **Data Processing**: Create train/validation splits with efficient DataLoaders
3. **Model Training**: Train TopK SAE with reconstruction + L1 losses
4. **Evaluation**: Comprehensive metrics including sparsity, reconstruction quality
5. **Analysis**: Feature analysis and interpretability metrics

## Project Structure

```
sae/
├── src/topk_sae/
│   ├── models/
│   │   └── topk_sae.py          # TopK SAE implementation
│   ├── data/
│   │   └── data_loader.py       # Activation data loading with nnsight
│   ├── training/
│   │   └── trainer.py           # Training loop and utilities
│   ├── config/
│   │   └── config.py            # Configuration management
│   └── utils/
│       └── ...                  # Utility functions
├── scripts/
│   └── train_topk_sae.py        # Main training script
├── configs/
│   ├── gemma2_2b.yaml          # Default Gemma2-2b config
│   └── quick_test.yaml         # Quick test config
├── tests/
│   └── test_topk_sae.py        # Comprehensive tests
└── pyproject.toml              # Project configuration
```

## Configuration Options

### Model Configuration

- `input_dim`: Input activation dimension (auto-detected)
- `hidden_dim`: Hidden layer dimension (typically 4-8x input_dim)
- `k`: Number of top activations to keep
- `tied_weights`: Whether to tie encoder/decoder weights
- `normalize_decoder`: Whether to normalize decoder weights

### Data Configuration

- `model_name`: HuggingFace model name (e.g., "google/gemma-2-2b")
- `layer_name`: Layer to extract activations from (e.g., "model.layers.15")
- `dataset_name`: Text dataset name (e.g., "openwebtext")
- `num_samples`: Number of samples to use for training
- `max_length`: Maximum sequence length for tokenization

### Training Configuration

- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer
- `batch_size`: Training batch size
- `l1_lambda`: L1 sparsity penalty coefficient
- `eval_every`: Evaluation frequency (steps)
- `save_every`: Checkpoint saving frequency (steps)

## Usage Examples

### Basic Training

```python
from topk_sae.config.config import get_gemma2_2b_config
from topk_sae.models.topk_sae import TopKSAE
from topk_sae.data.data_loader import ActivationDataLoader
from topk_sae.training.trainer import SAETrainer

# Load configuration
config = get_gemma2_2b_config()

# Set up data
data_loader = ActivationDataLoader(
    model_name=config.data.model_name,
    layer_name=config.data.layer_name
)

# Create dataset
dataset = data_loader.create_dataset_from_text(
    dataset_name=config.data.dataset_name,
    num_samples=10000
)

# Create model
model = TopKSAE(
    input_dim=dataset.activations.shape[-1],
    hidden_dim=config.model.hidden_dim,
    k=config.model.k
)

# Train
trainer = SAETrainer(model=model, train_dataloader=train_loader)
trainer.train(num_epochs=5)
```

### Activation Analysis

```python
# Load trained model
model = TopKSAE.load_checkpoint("checkpoints/best_model.pt")

# Analyze features
analysis = trainer.analyze_features(val_loader)
print(f"Dead features: {analysis['num_dead_features']}")
print(f"Sparsity: {analysis['sparsity']:.3f}")
print(f"Reconstruction quality: {analysis['cosine_similarity']:.3f}")
```

### Custom Layer Analysis

```python
# Extract activations from different layers
layers_to_analyze = [
    "model.layers.5",   # Early layer
    "model.layers.15",  # Middle layer  
    "model.layers.25",  # Late layer
]

for layer in layers_to_analyze:
    data_loader = ActivationDataLoader(
        model_name="google/gemma-2-2b",
        layer_name=layer
    )
    # ... train SAE for this layer
```

## Command Line Interface

The main training script supports various options:

```bash
# Basic training
python scripts/train_topk_sae.py

# With configuration file
python scripts/train_topk_sae.py --config configs/gemma2_2b.yaml

# Quick test
python scripts/train_topk_sae.py --quick-test

# Resume from checkpoint
python scripts/train_topk_sae.py --load-checkpoint checkpoints/best_model.pt

# Evaluation only
python scripts/train_topk_sae.py --eval-only --load-checkpoint checkpoints/best_model.pt

# Save/load activations
python scripts/train_topk_sae.py --save-activations activations.pt
python scripts/train_topk_sae.py --load-activations activations.pt
```

## Monitoring and Logging

### TensorBoard

View training metrics in real-time:

```bash
tensorboard --logdir logs
```

### Weights & Biases

Set `use_wandb: true` in your configuration and set your API key:

```bash
export WANDB_API_KEY=your_api_key
python scripts/train_topk_sae.py --config configs/gemma2_2b.yaml
```

## Performance Tips

### Memory Optimization

- Use smaller batch sizes if you encounter OOM errors
- Enable gradient checkpointing for very large models
- Use mixed precision training with `dtype: float16`

### Speed Optimization

- Use multiple GPUs with `device_map="auto"`
- Pre-collect and save activations for multiple training runs
- Use streaming datasets for very large datasets

### Quality Optimization

- Tune the `k` parameter for desired sparsity level
- Adjust `l1_lambda` for sparsity-reconstruction trade-off
- Use learning rate scheduling for better convergence

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_topk_sae.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes with proper type annotations and tests
4. Run tests: `pytest tests/`
5. Run linting: `ruff check src/`
6. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{topk_sae_gemma2,
  title={TopK Sparse Autoencoder for Gemma2-2b},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/topk-sae}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [nnsight](https://github.com/ndif-team/nnsight) for neural network interpretation
- [Gemma](https://ai.google.dev/gemma) team for the language model
- [Anthropic](https://www.anthropic.com/) for sparse autoencoder research