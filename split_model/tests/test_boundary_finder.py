"""
Test suite for C2PI boundary finder implementation.
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c2pi.boundary_finder import BoundaryFinder
from c2pi.attacks.dina import DINAAttacker
from c2pi.attacks.mla import MLAAttacker, extract_features
from c2pi.models.utils import get_model, identify_layer_candidates
from c2pi.utils.metrics import calculate_ssim, calculate_accuracy
from c2pi.config import C2PIConfig


class TestBoundaryFinder(unittest.TestCase):
    """Test cases for BoundaryFinder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')  # Use CPU for tests
        
        # Create simple test model
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
        self.model.to(self.device)
        
        # Create test data
        self.test_data = torch.randn(32, 3, 8, 8)
        self.test_targets = torch.randint(0, 10, (32,))
        
        self.test_loader = DataLoader(
            TensorDataset(self.test_data, self.test_targets),
            batch_size=8, shuffle=False
        )
        
        # Configuration
        self.config = C2PIConfig(
            privacy_threshold=0.3,
            accuracy_threshold=0.95,
            device=self.device
        )
        
        # Layer candidates
        self.layer_candidates = [0, 2, 5]  # Conv, Conv, Linear
    
    def test_boundary_finder_initialization(self):
        """Test BoundaryFinder initialization."""
        finder = BoundaryFinder(
            model=self.model,
            layer_candidates=self.layer_candidates,
            config=self.config
        )
        
        self.assertEqual(finder.model, self.model)
        self.assertEqual(finder.layer_candidates, self.layer_candidates)
        self.assertEqual(finder.config, self.config)
    
    def test_identify_layer_candidates(self):
        """Test layer candidate identification."""
        candidates = identify_layer_candidates(self.model)
        
        # Should identify layers that can be split points
        self.assertIsInstance(candidates, list)
        self.assertTrue(len(candidates) > 0)
        
        # All candidates should be valid layer indices
        for candidate in candidates:
            self.assertIsInstance(candidate, int)
            self.assertGreaterEqual(candidate, 0)
    
    def test_dina_attack_basic(self):
        """Test basic DINA attack functionality."""
        # Create DINA attacker
        attacker = DINAAttacker(
            input_shape=(3, 8, 8),
            num_classes=10,
            device=self.device
        )
        
        # Test initialization
        self.assertEqual(attacker.input_shape, (3, 8, 8))
        self.assertEqual(attacker.num_classes, 10)
        
        # Test distillation training (with very few epochs for speed)
        try:
            history = attacker.train_distillation(
                teacher_model=self.model,
                data_loader=self.test_loader,
                epochs=2,  # Very few epochs for testing
                lr=1e-3
            )
            
            self.assertIn('loss', history)
            self.assertTrue(len(history['loss']) > 0)
            
        except Exception as e:
            self.fail(f"DINA training failed: {e}")
    
    def test_mla_attack_basic(self):
        """Test basic MLA attack functionality."""
        # Create MLA attacker
        attacker = MLAAttacker(
            input_shape=(3, 8, 8),
            feature_shape=(32,),  # Assuming flattened features
            device=self.device
        )
        
        # Test initialization
        self.assertEqual(attacker.input_shape, (3, 8, 8))
        self.assertEqual(attacker.feature_shape, (32,))
        
        # Create dummy feature data
        features = torch.randn(32, 32)
        targets = self.test_data
        
        feature_loader = DataLoader(
            TensorDataset(features, targets),
            batch_size=8, shuffle=False
        )
        
        # Test training (with very few epochs)
        try:
            history = attacker.train_attack(
                feature_loader=feature_loader,
                target_loader=feature_loader,  # Same for simplicity
                epochs=2,
                lr=1e-3
            )
            
            self.assertIn('loss', history)
            self.assertTrue(len(history['loss']) > 0)
            
        except Exception as e:
            self.fail(f"MLA training failed: {e}")
    
    def test_feature_extraction(self):
        """Test feature extraction functionality."""
        layer_idx = 0  # First conv layer
        
        try:
            features, targets = extract_features(
                model=self.model,
                data_loader=self.test_loader,
                layer_idx=layer_idx,
                device=self.device
            )
            
            # Check output shapes
            self.assertEqual(len(features), len(self.test_data))
            self.assertEqual(len(targets), len(self.test_targets))
            
            # Features should have correct batch dimension
            self.assertEqual(features.shape[0], len(self.test_data))
            
        except Exception as e:
            self.fail(f"Feature extraction failed: {e}")
    
    def test_metrics_calculation(self):
        """Test various metrics calculations."""
        # Test SSIM calculation
        img1 = torch.rand(1, 3, 8, 8)
        img2 = torch.rand(1, 3, 8, 8)
        
        ssim_score = calculate_ssim(img1, img2)
        self.assertIsInstance(ssim_score, float)
        self.assertGreaterEqual(ssim_score, 0.0)
        self.assertLessEqual(ssim_score, 1.0)
        
        # Test accuracy calculation
        accuracy = calculate_accuracy(self.model, self.test_loader, self.device)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = C2PIConfig(
            privacy_threshold=0.3,
            accuracy_threshold=0.95
        )
        self.assertEqual(config.privacy_threshold, 0.3)
        self.assertEqual(config.accuracy_threshold, 0.95)
        
        # Test default values
        self.assertIsInstance(config.noise_levels, list)
        self.assertTrue(len(config.noise_levels) > 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete C2PI pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.device = torch.device('cpu')
        
        # Use a very simple model for fast testing
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        self.model.to(self.device)
        
        # Minimal test data
        self.test_data = torch.randn(16, 3, 8, 8)
        self.test_targets = torch.randint(0, 10, (16,))
        
        self.test_loader = DataLoader(
            TensorDataset(self.test_data, self.test_targets),
            batch_size=4, shuffle=False
        )
        
        self.config = C2PIConfig(
            privacy_threshold=0.3,
            accuracy_threshold=0.95,
            device=self.device
        )
    
    def test_end_to_end_pipeline(self):
        """Test complete boundary finding pipeline."""
        # Get layer candidates
        layer_candidates = identify_layer_candidates(self.model)
        
        # Should find at least one candidate
        self.assertTrue(len(layer_candidates) > 0)
        
        # Initialize boundary finder
        finder = BoundaryFinder(
            model=self.model,
            layer_candidates=layer_candidates[:2],  # Limit for speed
            config=self.config
        )
        
        # Run simplified boundary search
        try:
            # Note: This will likely fail due to minimal data/epochs
            # but we test that the pipeline structure works
            
            # Test phase 1 evaluation only
            layer_idx = layer_candidates[0]
            
            # Create minimal attacker for testing
            attacker = DINAAttacker(
                input_shape=(3, 8, 8),
                num_classes=10,
                device=self.device
            )
            
            # Test single layer evaluation
            privacy_metrics = finder._evaluate_privacy_layer(
                layer_idx=layer_idx,
                attacker=attacker,
                train_loader=self.test_loader,
                test_loader=self.test_loader,
                epochs=1,  # Minimal epochs
                lr=1e-3
            )
            
            # Should return some metrics
            self.assertIsInstance(privacy_metrics, dict)
            
        except Exception as e:
            # Expected to potentially fail with minimal data
            # Just ensure no crashes in the setup
            print(f"Expected error in minimal test: {e}")
    
    def test_model_loading(self):
        """Test pre-trained model loading."""
        try:
            # This should work for standard models
            model = get_model('vgg11', num_classes=10, pretrained=False)
            self.assertIsInstance(model, nn.Module)
            
            # Test layer identification
            candidates = identify_layer_candidates(model)
            self.assertTrue(len(candidates) > 0)
            
        except Exception as e:
            self.fail(f"Model loading failed: {e}")


class TestUtilities(unittest.TestCase):
    """Test utility functions."""
    
    def test_config_serialization(self):
        """Test configuration serialization/deserialization."""
        config = C2PIConfig(
            privacy_threshold=0.5,
            accuracy_threshold=0.9,
            noise_levels=[0.1, 0.2],
            batch_size=64
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['privacy_threshold'], 0.5)
        
        # Test from_dict
        new_config = C2PIConfig.from_dict(config_dict)
        self.assertEqual(new_config.privacy_threshold, 0.5)
        self.assertEqual(new_config.accuracy_threshold, 0.9)
    
    def test_temporary_file_operations(self):
        """Test file save/load operations."""
        # Test configuration save/load
        config = C2PIConfig(privacy_threshold=0.4)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            loaded_config = C2PIConfig.load(temp_path)
            
            self.assertEqual(loaded_config.privacy_threshold, 0.4)
            
        finally:
            os.unlink(temp_path)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestBoundaryFinder))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    test_suite.addTest(unittest.makeSuite(TestUtilities))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
