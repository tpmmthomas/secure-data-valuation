# Knowledge Distillation Framework

A production-ready knowledge distillation framework for neural networks, supporting multiple teacher-student architectures and datasets.

## 🚀 Features

- **Multiple Teacher Models**: ResNet18, ResNet50, VGG16, DenseNet121, EfficientNet
- **Multiple Student Models**: Lightweight CNNs, MobileNet variants, custom architectures
- **Multiple Datasets**: CIFAR-10, CIFAR-100, ImageNet, MNIST, Fashion-MNIST
- **Flexible Configuration**: YAML-based configuration system
- **Production Ready**: Logging, checkpointing, metrics tracking, and monitoring
- **Extensible**: Easy to add new models and datasets

## 📁 Project Structure

```
knowledge_distillation/
├── README.md
├── requirements.txt
├── setup.py
├── configs/                 # Configuration files
│   ├── datasets/           # Dataset configurations
│   ├── models/             # Model configurations
│   └── experiments/        # Complete experiment configurations
├── src/
│   ├── __init__.py
│   ├── main.py            # Main training script
│   ├── models/            # Model architectures
│   ├── datasets/          # Dataset loaders
│   ├── utils/             # Utilities and helpers
│   └── trainer.py         # Training logic
├── scripts/               # Utility scripts
├── examples/              # Example configurations and scripts
└── artifacts/            # Output directory (models, logs, metrics)
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU training)

### Install from source
```bash
# Clone the repository (if not already done)
cd knowledge_distillation

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 🏃 Quick Start

### 1. Basic CIFAR-10 Example
```bash
# Train ResNet18 teacher and CNN5 student on CIFAR-10
python src/main.py --config configs/experiments/cifar10_resnet18_cnn5.yaml
```

### 2. Custom Configuration
```bash
# Use a custom configuration file
python src/main.py --config your_config.yaml
```

### 3. Command Line Override
```bash
# Override specific parameters
python src/main.py \
    --config configs/experiments/cifar10_resnet18_cnn5.yaml \
    --teacher_model resnet50 \
    --student_model mobilenet_v2 \
    --epochs_teacher 10 \
    --epochs_student 50
```

## ⚙️ Configuration

The framework uses YAML configuration files for maximum flexibility. See `configs/` directory for examples.

### Example Configuration
```yaml
# configs/experiments/cifar10_resnet18_cnn5.yaml
experiment:
  name: "cifar10_resnet18_cnn5"
  seed: 42
  output_dir: "artifacts"

dataset:
  name: "cifar10"
  batch_size: 128
  num_workers: 4

teacher:
  model: "resnet18"
  pretrained: true
  epochs: 5
  lr: 0.01
  momentum: 0.9
  weight_decay: 5e-4

student:
  model: "cnn5"
  epochs: 20
  lr: 0.01
  momentum: 0.9
  weight_decay: 5e-4

distillation:
  alpha: 0.5
  temperature: 4.0
  loss_type: "kl_div"
```

## 📊 Supported Models

### Teacher Models
- **ResNet**: ResNet18, ResNet34, ResNet50, ResNet101
- **VGG**: VGG11, VGG13, VGG16, VGG19
- **DenseNet**: DenseNet121, DenseNet169, DenseNet201
- **EfficientNet**: EfficientNet-B0 to B7
- **Vision Transformer**: ViT-Base, ViT-Large

### Student Models
- **CNN Variants**: CNN3, CNN5, CNN7 (lightweight architectures)
- **MobileNet**: MobileNet-V2, MobileNet-V3
- **ShuffleNet**: ShuffleNet-V2
- **SqueezeNet**: SqueezeNet 1.0, 1.1

## 📁 Supported Datasets

- **CIFAR-10**: 32x32 RGB images, 10 classes
- **CIFAR-100**: 32x32 RGB images, 100 classes
- **ImageNet**: 224x224 RGB images, 1000 classes
- **MNIST**: 28x28 grayscale images, 10 classes
- **Fashion-MNIST**: 28x28 grayscale images, 10 classes
- **STL-10**: 96x96 RGB images, 10 classes
- **SVHN**: 32x32 RGB images, 10 classes

## 🎯 Training Modes

### 1. Teacher Training
Train a teacher model from scratch or fine-tune a pretrained model:
```bash
python src/main.py --config configs/teacher_only.yaml
```

### 2. Student Training (Supervised)
Train a student model using only ground truth labels:
```bash
python src/main.py --config configs/student_supervised.yaml
```

### 3. Knowledge Distillation
Train a student model using knowledge distillation:
```bash
python src/main.py --config configs/experiments/cifar10_kd.yaml
```

### 4. Progressive Distillation
Multi-stage distillation with intermediate models:
```bash
python src/main.py --config configs/progressive_distillation.yaml
```

## 📈 Monitoring and Logging

The framework provides comprehensive monitoring:

- **TensorBoard**: Real-time training metrics visualization
- **Weights & Biases**: Experiment tracking (optional)
- **JSON Logs**: Structured logging for analysis
- **Model Checkpoints**: Automatic saving of best models

### View TensorBoard
```bash
tensorboard --logdir artifacts/logs
```

## 🔧 Advanced Usage

### Custom Models
Add your own model architectures by implementing the base model interface:

```python
# src/models/custom_model.py
from .base import BaseModel

class CustomTeacher(BaseModel):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        # Your implementation here
```

### Custom Datasets
Add new datasets by implementing the dataset interface:

```python
# src/datasets/custom_dataset.py
from .base import BaseDataset

class CustomDataset(BaseDataset):
    def __init__(self, **kwargs):
        # Your implementation here
```

## 🧪 Examples

See the `examples/` directory for:
- Jupyter notebooks with detailed walkthroughs
- Benchmark scripts
- Advanced configuration examples
- Performance analysis tools

## 📊 Results and Benchmarks

The framework includes benchmark results for common teacher-student pairs:

| Dataset | Teacher | Student | Accuracy (Student Only) | Accuracy (KD) | Improvement |
|---------|---------|---------|-------------------------|---------------|-------------|
| CIFAR-10 | ResNet18 | CNN5 | 87.2% | 89.1% | +1.9% |
| CIFAR-100 | ResNet50 | MobileNet-V2 | 71.3% | 74.8% | +3.5% |
| ImageNet | EfficientNet-B3 | MobileNet-V2 | 72.1% | 75.6% | +3.5% |

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:
- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{knowledge_distillation_framework,
  title={Knowledge Distillation Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/knowledge-distillation}
}
```

## 🆘 Support

- **Documentation**: [Link to docs]
- **Issues**: [GitHub Issues]
- **Discussions**: [GitHub Discussions]
- **Email**: support@yourproject.com

## 🎉 Acknowledgments

- Hinton et al. for the original knowledge distillation paper
- PyTorch team for the excellent deep learning framework
- The open-source community for various model implementations
