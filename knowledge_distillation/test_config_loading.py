#!/usr/bin/env python3
"""Test script to verify config loading works correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config, KnowledgeDistillationConfig

def test_config_loading():
    """Test loading a config file."""
    try:
        config_path = "configs/experiments/cifar10_resnet18_cnn5.yaml"
        print(f"Loading config from: {config_path}")
        
        config = load_config(config_path)
        
        print(f"Config type: {type(config)}")
        print(f"Is KnowledgeDistillationConfig? {isinstance(config, KnowledgeDistillationConfig)}")
        
        if hasattr(config, 'experiment'):
            print(f"Experiment name: {config.experiment.name}")
            print(f"Experiment type: {type(config.experiment)}")
        else:
            print("Config has no 'experiment' attribute")
            print(f"Config attributes: {dir(config)}")
            
        print("✓ Config loading test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Config loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_config_loading()
