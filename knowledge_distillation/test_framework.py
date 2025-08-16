#!/usr/bin/env python3
"""Simple test script to validate the framework setup."""

import sys
from pathlib import Path

# Add the parent directory to path so we can import src as a package
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.config import KnowledgeDistillationConfig, load_config
        print("✓ Config module imported successfully")
        
        from src.models import get_available_models, create_model
        print("✓ Models module imported successfully")
        
        from src.datasets import DATASETS, create_dataset
        print("✓ Datasets module imported successfully")
        
        from src.utils import set_seed, get_device
        print("✓ Utils module imported successfully")
        
        from src.trainer import KnowledgeDistillationTrainer
        print("✓ Trainer module imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_models():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from src.models import get_available_models, create_model
        
        available = get_available_models()
        print(f"Available teachers: {available['teachers'][:3]}...")  # Show first 3
        print(f"Available students: {available['students'][:3]}...")   # Show first 3
        
        # Test teacher creation
        teacher = create_model('teacher', 'resnet18', num_classes=10, pretrained=False)
        print(f"✓ Created teacher: {teacher.__class__.__name__}")
        
        # Test student creation
        student = create_model('student', 'cnn5', num_classes=10)
        print(f"✓ Created student: {student.__class__.__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_datasets():
    """Test dataset creation."""
    print("\nTesting dataset creation...")
    
    try:
        from src.datasets import DATASETS, create_dataset
        
        print(f"Available datasets: {list(DATASETS.keys())}")
        
        # Test dataset creation
        dataset = create_dataset(
            'mnist', batch_size=32, num_workers=0, download=False
        )
        
        train_loader, test_loader, val_loader = dataset.get_dataloaders()
        dataset_info = dataset.get_dataset_info()
        
        print(f"✓ Created dataset loader: {dataset_info['name']}")
        print(f"  - Num classes: {dataset.num_classes}")
        print(f"  - Input shape: {dataset.input_shape}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False


def test_config():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from src.config import KnowledgeDistillationConfig
        
        # Test basic config creation with default values
        config = KnowledgeDistillationConfig()
        
        print(f"✓ Configuration created successfully")
        print(f"  - Experiment name: {config.experiment.name}")
        print(f"  - Dataset: {config.dataset.name}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=== Knowledge Distillation Framework Test ===\n")
    
    tests = [
        test_imports,
        test_models,
        test_datasets,
        test_config
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 All tests passed! Framework is ready to use.")
        
        print("\n=== Next Steps ===")
        print("1. Try running a basic experiment:")
        print("   python src/main.py --config configs/experiments/cifar10_resnet18_cnn5.yaml")
        print("\n2. Or start with a smaller test:")
        print("   python src/main.py --config configs/experiments/mnist_vgg16_cnn3.yaml")
        
    else:
        print("❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
