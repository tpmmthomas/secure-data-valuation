# C2PI: Crypto-Clear Two-Party Neural Network Private Inference

This repository implements Algorithm 1 from the paper "C2PI: An Efficient Crypto-Clear Two-Party Neural Network Private Inference" for finding optimal boundary layers in neural networks for split inference.

## Overview

C2PI introduces a novel approach to private inference by partitioning neural networks into:
- **Crypto layers**: Executed with expensive MPC protocols for cryptographic privacy
- **Clear layers**: Executed in plaintext after empirical privacy verification

The key innovation is using **DINA (Distillation-based Inverse-Network Attack)** to find the boundary layer where privacy attacks fail, enabling significant computational savings.

## Features

- 🔍 **Boundary Layer Detection**: Implements Algorithm 1 to find optimal crypto/clear partitioning
- 🛡️ **Enhanced Privacy Attacks**: DINA with distillation points for robust privacy evaluation
- 📊 **Multiple Model Support**: VGG, ResNet, AlexNet architectures
- 🎯 **Configurable Privacy**: Adjustable SSIM thresholds and noise levels
- 📈 **Comprehensive Evaluation**: Performance metrics and visualization tools

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from c2pi import BoundaryFinder, Config

# Configure experiment
config = Config(
    dataset="CIFAR10",
    model_name="vgg16",
    ssim_threshold=0.3,
    noise_lambda=0.1,
    accuracy_tolerance=0.025
)

# Find boundary layer
finder = BoundaryFinder(config)
boundary_layer, results = finder.find_boundary()

print(f"Optimal boundary: Layer {boundary_layer}")
print(f"Privacy preserved: {results['privacy_preserved']}")
print(f"Accuracy drop: {results['accuracy_drop']:.2%}")
```

## Algorithm 1 Implementation

The core boundary search algorithm consists of two phases:

### Phase 1: Privacy Evaluation
```python
l_prime = n - 1
avg_ssim = IDPA(l_prime)  # Run DINA attack
while avg_ssim < ssim_threshold:
    l_prime = l_prime - 1
    avg_ssim = IDPA(l_prime)
l_prime = l_prime + 1  # First layer where attack succeeds
```

### Phase 2: Accuracy Validation
```python
n_acc = accuracy_with_noise(l_prime, noise_lambda)
while n_acc < accuracy_threshold:
    l_prime = l_prime + 1
    n_acc = accuracy_with_noise(l_prime, noise_lambda)
return l_prime  # Boundary layer
```

## Configuration

Key parameters in `config.py`:

```python
@dataclass
class Config:
    # Dataset and Model
    dataset: str = "CIFAR10"  # CIFAR10, CIFAR100, ImageNet
    model_name: str = "vgg16"  # vgg16, vgg19, resnet50, alexnet
    
    # Privacy Parameters
    ssim_threshold: float = 0.3  # IDPA failure threshold
    noise_lambda: float = 0.1    # Noise magnitude for defense
    accuracy_tolerance: float = 0.025  # Max accuracy drop (2.5%)
    
    # DINA Training
    dina_epochs: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 128
    
    # Loss coefficients: α₀=1, α₁=3, αⱼ=2×αⱼ₋₁
    alpha_base: List[float] = None  # Auto-computed
```

## Project Structure

```
split_model/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── config.py                # Configuration classes
├── main.py                  # Main execution script
├── c2pi/
│   ├── __init__.py
│   ├── boundary_finder.py   # Algorithm 1 implementation
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── dina.py         # DINA attack implementation
│   │   ├── mla.py          # Maximum Likelihood Attack
│   │   └── utils.py        # Attack utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vgg.py          # VGG architectures
│   │   ├── resnet.py       # ResNet architectures
│   │   └── utils.py        # Model utilities
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py     # Dataset loading
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py      # SSIM, accuracy metrics
│       ├── noise.py        # Noise injection
│       └── visualization.py # Plotting tools
├── examples/
│   ├── basic_usage.py      # Simple example
│   ├── comparative_study.py # Compare different models
│   └── ablation_study.py   # Parameter sensitivity
├── tests/
│   ├── __init__.py
│   ├── test_boundary_finder.py
│   ├── test_dina.py
│   └── test_models.py
└── docs/
    ├── algorithm.md        # Detailed algorithm explanation
    ├── api_reference.md    # API documentation
    └── examples.md         # Usage examples
```

## Usage Examples

### Basic Boundary Finding

```python
from c2pi import BoundaryFinder, Config

# Default configuration for CIFAR-10 + VGG16
config = Config()
finder = BoundaryFinder(config)

# Find boundary with privacy evaluation
results = finder.run_full_evaluation()

print("Results:")
print(f"  Boundary Layer: {results['boundary_layer']}")
print(f"  Crypto Layers: 0-{results['boundary_layer']}")
print(f"  Clear Layers: {results['boundary_layer']+1}-{results['total_layers']}")
print(f"  Privacy Preserved: {results['privacy_preserved']}")
print(f"  Accuracy: {results['accuracy']:.2%}")
print(f"  Speedup Estimate: {results['speedup_estimate']:.2f}x")
```

### Custom Privacy Thresholds

```python
# More conservative privacy (lower SSIM threshold)
config = Config(ssim_threshold=0.2)
finder = BoundaryFinder(config)
conservative_boundary = finder.find_boundary()

# More aggressive optimization (higher SSIM threshold)
config = Config(ssim_threshold=0.4)
finder = BoundaryFinder(config)
aggressive_boundary = finder.find_boundary()
```

### Comparative Study

```python
from c2pi.utils.visualization import plot_boundary_comparison

models = ["vgg16", "vgg19", "resnet50"]
boundaries = {}

for model in models:
    config = Config(model_name=model)
    finder = BoundaryFinder(config)
    boundaries[model] = finder.find_boundary()

plot_boundary_comparison(boundaries)
```

## Performance Results

Based on the paper's experimental results:

| Model | Dataset | Boundary Layer | Speedup (LAN) | Speedup (WAN) | Comm. Reduction |
|-------|---------|----------------|---------------|---------------|-----------------|
| VGG16 | CIFAR-10| 9              | 1.46x         | 1.71x         | 2.5x           |
| VGG19 | CIFAR-10| 9              | 1.46x         | 1.82x         | 2.75x          |
| VGG16 | CIFAR-100| 10            | 1.19x         | 1.0x          | 1.1x           |

## API Reference

### BoundaryFinder

Main class for finding optimal boundary layers.

```python
class BoundaryFinder:
    def __init__(self, config: Config)
    def find_boundary(self) -> Tuple[int, Dict]
    def evaluate_privacy(self, layer_idx: int) -> float
    def evaluate_accuracy(self, layer_idx: int) -> float
    def run_full_evaluation(self) -> Dict
```

### DINA Attack

Enhanced inverse network attack with distillation.

```python
class DINAAttack:
    def __init__(self, model: nn.Module, config: Config)
    def train(self, dataloader: DataLoader) -> None
    def attack(self, activations: torch.Tensor) -> torch.Tensor
    def evaluate_ssim(self, dataloader: DataLoader) -> float
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a Pull Request

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_boundary_finder.py -v

# Run with coverage
python -m pytest tests/ --cov=c2pi --cov-report=html
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhang2023c2pi,
  title={C2PI: An Efficient Crypto-Clear Two-Party Neural Network Private Inference},
  author={Zhang, Yuke and Chen, Dake and Kundu, Souvik and Liu, Haomei and Peng, Ruiheng and Beerel, Peter A.},
  journal={arXiv preprint arXiv:2304.13266},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original C2PI paper authors
- PyTorch team for the deep learning framework
- Open source contributors to privacy-preserving ML
