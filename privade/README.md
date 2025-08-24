# PrivaDE Split Model Module

This module provides functionality for splitting neural networks into client and server parts while preserving privacy using the DINA (Distillation-based Inverse Network Attack) algorithm.

## Features

- **Privacy-Preserving Model Splitting**: Automatically finds optimal boundary layers for split inference
- **DINA Attack Evaluation**: Uses state-of-the-art inverse network attacks to evaluate privacy
- **Flexible Configuration**: Customizable privacy thresholds and attack parameters
- **Direct Model/DataLoader Support**: Works directly with PyTorch DataLoaders and nn.Module objects

## Installation

```bash
pip install torch torchvision numpy scipy scikit-learn tqdm
```

## Usage

### Basic Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from privade.split import split_model, SplitConfig

# Your model and dataloader
model = your_model()  # nn.Module
data_loader = your_dataloader()  # DataLoader

# Configure splitting parameters
config = SplitConfig(
    privacy_threshold=0.8,  # Minimum privacy preservation rate
    attack_epochs=50,       # Epochs for DINA attack training
    device='auto'           # Auto-detect GPU/CPU
)

# Split the model
client_model, server_model, statistics = split_model(
    data_loader=data_loader,
    model=model,
    dataset_mean=[0.485, 0.456, 0.406],  # Your dataset normalization
    dataset_std=[0.229, 0.224, 0.225],
    config=config
)

print(f"Optimal boundary layer: {statistics['optimal_layer']}")
print(f"Privacy preserved rate: {statistics['privacy_preserved_rate']:.3f}")
```

### Function Signature

```python
def split_model(
    data_loader: DataLoader,
    model: nn.Module,
    dataset_mean: List[float] = [0.5, 0.5, 0.5],
    dataset_std: List[float] = [0.5, 0.5, 0.5],
    config: Optional[SplitConfig] = None
) -> Tuple[nn.Module, nn.Module, Dict[str, Any]]
```

### Parameters

- **data_loader**: PyTorch DataLoader containing the dataset for evaluation
- **model**: The neural network model to split (nn.Module)
- **dataset_mean**: Dataset normalization mean values (default: [0.5, 0.5, 0.5])
- **dataset_std**: Dataset normalization standard deviation values (default: [0.5, 0.5, 0.5])
- **config**: SplitConfig object with algorithm parameters (optional)

### Returns

A tuple containing:
1. **client_model**: The client-side model (nn.Module)
2. **server_model**: The server-side model (nn.Module)  
3. **statistics**: Dictionary with split analysis results

### Split Statistics

The statistics dictionary contains:
- `optimal_layer`: Index of the optimal boundary layer
- `privacy_preserved_rate`: Privacy preservation rate (0-1)
- `attack_success_rate`: DINA attack success rate (0-1)
- `avg_ssim`: Average SSIM score between original and reconstructed images
- `layer_results`: Detailed results for all evaluated layers
- `config`: Configuration parameters used

### Configuration Options

```python
@dataclass
class SplitConfig:
    privacy_threshold: float = 0.9        # Minimum privacy preservation rate
    ssim_threshold: float = 0.3           # SSIM threshold for privacy evaluation
    learning_rate: float = 1e-3           # Learning rate for attack training
    device: Union[str, torch.device] = 'auto'  # Device ('auto', 'cpu', 'cuda')
    attack_epochs: int = 50               # Epochs for DINA attack training
    attack_patience: int = 10             # Early stopping patience
    max_samples: Optional[int] = None     # Limit dataset size for evaluation
```

## Algorithm

The split model algorithm uses the C2PI (Crypto-Clear Privacy-preserving Inference) approach:

1. **Candidate Layer Identification**: Identifies potential boundary layers in the model
2. **Privacy Evaluation**: For each candidate layer, trains a DINA attack to reconstruct input images
3. **Privacy Metrics**: Calculates privacy preservation rate based on reconstruction quality
4. **Optimal Selection**: Selects the layer with the highest privacy preservation rate above the threshold

## Examples

See `example_split_usage.py` for a complete working example with CIFAR-10.

## Performance Considerations

- **Dataset Size**: Use `max_samples` in config to limit evaluation dataset size for faster processing
- **Attack Epochs**: Reduce `attack_epochs` for faster evaluation (may reduce privacy evaluation accuracy)
- **GPU Usage**: The algorithm automatically uses GPU if available for faster training

## Privacy Metrics

- **Privacy Preserved Rate**: Fraction of samples where DINA attack fails to reconstruct the input
- **Attack Success Rate**: Fraction of samples successfully reconstructed by the attack
- **SSIM Score**: Structural Similarity Index between original and reconstructed images

## Notes

- The algorithm assumes models with sequential layer structure
- Best suited for CNN architectures with Conv2d and Linear layers
- Privacy evaluation is computational intensive and may take time depending on model size and dataset
