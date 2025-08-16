# Knowledge Distillation Quick Start

This notebook demonstrates how to use the Knowledge Distillation Framework for training neural networks.

## Setup

```python
import sys
sys.path.append('../src')

from config import KnowledgeDistillationConfig, load_config
from trainer import KnowledgeDistillationTrainer
from models import get_available_models
from datasets import DATASETS
import torch

# Check available models and datasets
print("Available models:", get_available_models())
print("Available datasets:", list(DATASETS.keys()))
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

## Configuration

```python
# Load a pre-defined configuration
config = load_config('../configs/experiments/cifar10_resnet18_cnn5.yaml')

# Or create a custom configuration
from config import (
    KnowledgeDistillationConfig, ExperimentConfig, 
    DatasetConfig, TeacherConfig, StudentConfig, DistillationConfig
)

custom_config = KnowledgeDistillationConfig(
    experiment=ExperimentConfig(
        name="my_custom_experiment",
        seed=42,
        output_dir="artifacts/custom"
    ),
    dataset=DatasetConfig(
        name="cifar10",
        batch_size=64,
        num_workers=2
    ),
    teacher=TeacherConfig(
        name="resnet18",
        pretrained=True,
        epochs=3,
        lr=0.01
    ),
    student=StudentConfig(
        name="cnn5",
        epochs=10,
        lr=0.01
    ),
    distillation=DistillationConfig(
        alpha=0.5,
        temperature=4.0
    )
)
```

## Training

```python
# Initialize trainer
trainer = KnowledgeDistillationTrainer(config)

# Run full training pipeline
results = trainer.train()

# Or train components separately
# teacher_results = trainer.train_teacher()
# student_results = trainer.train_student_kd()
```

## Results Analysis

```python
# Plot training curves
from utils import plot_training_curves, plot_model_comparison

plot_training_curves(results, save_path="training_curves.png")

# Compare model performance
teacher_acc = results["final"]["teacher_test"]["acc"]
student_acc = results["final"]["student_kd_test"]["acc"]

plot_model_comparison(teacher_acc, 0, student_acc, save_path="comparison.png")

print(f"Teacher accuracy: {teacher_acc:.2f}%")
print(f"Student accuracy: {student_acc:.2f}%")
```

## Model Evaluation

```python
# Evaluate on test set
test_results = trainer._evaluate(trainer.student, trainer.test_loader)
print(f"Final test accuracy: {test_results['acc']:.2f}%")

# Model size comparison
teacher_params = sum(p.numel() for p in trainer.teacher.parameters())
student_params = sum(p.numel() for p in trainer.student.parameters())

print(f"Teacher parameters: {teacher_params:,}")
print(f"Student parameters: {student_params:,}")
print(f"Compression ratio: {teacher_params/student_params:.2f}x")
```

## Advanced Usage

### Custom Models

```python
# Add your own model to the framework
from models.base import StudentModel
import torch.nn as nn

class MyCustomStudent(StudentModel):
    def _initialize_model(self, **kwargs):
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(32, self.num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Register the model
from models.students import STUDENT_MODELS
STUDENT_MODELS['my_custom_student'] = MyCustomStudent
```

### Temperature Analysis

```python
# Test different temperatures
temperatures = [1.0, 2.0, 4.0, 8.0, 16.0]
results_by_temp = {}

for temp in temperatures:
    # Update config
    config.distillation.temperature = temp
    config.experiment.name = f"temp_analysis_{temp}"
    config.experiment.output_dir = f"artifacts/temp_{temp}"
    
    # Train
    trainer = KnowledgeDistillationTrainer(config)
    result = trainer.train()
    
    results_by_temp[temp] = result["final"]["student_kd_test"]["acc"]

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(list(results_by_temp.keys()), list(results_by_temp.values()), 'o-')
plt.xlabel('Temperature')
plt.ylabel('Student Accuracy (%)')
plt.title('Effect of Temperature on Knowledge Distillation')
plt.grid(True)
plt.show()
```

## Export and Deployment

```python
# Export student model to ONNX
import torch.onnx

dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(
    trainer.student,
    dummy_input,
    "student_model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)

# Save PyTorch model
torch.save(trainer.student.state_dict(), "student_model.pt")
```
