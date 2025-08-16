"""
Performance benchmarks for C2PI implementation.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import psutil
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c2pi.boundary_finder import BoundaryFinder
from c2pi.attacks.dina import DINAAttacker
from c2pi.attacks.mla import MLAAttacker, extract_features
from c2pi.models.utils import get_model, identify_layer_candidates
from c2pi.data.loaders import get_cifar10_loaders
from c2pi.config import C2PIConfig


class PerformanceBenchmark:
    """Benchmark C2PI performance across different configurations."""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
    
    def benchmark_model_loading(self, model_names=['vgg11', 'resnet18']):
        """Benchmark model loading times."""
        print("Benchmarking model loading...")
        
        results = {}
        for model_name in model_names:
            start_time = time.time()
            
            model = get_model(model_name, num_classes=10, pretrained=False)
            model = model.to(self.device)
            
            end_time = time.time()
            
            # Get model size
            param_count = sum(p.numel() for p in model.parameters())
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            results[model_name] = {
                'loading_time': end_time - start_time,
                'parameters': param_count,
                'size_mb': model_size_mb
            }
            
            print(f"  {model_name}: {end_time - start_time:.3f}s, {param_count:,} params, {model_size_mb:.1f}MB")
        
        self.results['model_loading'] = results
        return results
    
    def benchmark_layer_identification(self, model_names=['vgg11', 'resnet18']):
        """Benchmark layer candidate identification."""
        print("Benchmarking layer identification...")
        
        results = {}
        for model_name in model_names:
            model = get_model(model_name, num_classes=10, pretrained=False)
            
            start_time = time.time()
            candidates = identify_layer_candidates(model)
            end_time = time.time()
            
            results[model_name] = {
                'identification_time': end_time - start_time,
                'num_candidates': len(candidates),
                'candidates': candidates
            }
            
            print(f"  {model_name}: {end_time - start_time:.4f}s, {len(candidates)} candidates")
        
        self.results['layer_identification'] = results
        return results
    
    def benchmark_feature_extraction(self, model_name='vgg11', data_sizes=[100, 500, 1000]):
        """Benchmark feature extraction for different data sizes."""
        print("Benchmarking feature extraction...")
        
        model = get_model(model_name, num_classes=10, pretrained=False)
        model = model.to(self.device)
        model.eval()
        
        candidates = identify_layer_candidates(model)
        test_layer = candidates[len(candidates)//2]  # Middle layer
        
        results = {}
        
        for data_size in data_sizes:
            # Create test data
            test_data = torch.randn(data_size, 3, 32, 32)
            test_targets = torch.randint(0, 10, (data_size,))
            test_loader = DataLoader(
                TensorDataset(test_data, test_targets),
                batch_size=64, shuffle=False
            )
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            features, targets = extract_features(model, test_loader, test_layer, self.device)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            results[data_size] = {
                'extraction_time': end_time - start_time,
                'memory_usage_mb': end_memory - start_memory,
                'feature_shape': features.shape,
                'throughput_samples_per_sec': data_size / (end_time - start_time)
            }
            
            print(f"  {data_size} samples: {end_time - start_time:.3f}s, "
                  f"{end_memory - start_memory:.1f}MB, "
                  f"{data_size / (end_time - start_time):.1f} samples/sec")
        
        self.results['feature_extraction'] = results
        return results
    
    def benchmark_attack_training(self, attack_types=['dina'], epochs_list=[5, 10, 20]):
        """Benchmark attack training times."""
        print("Benchmarking attack training...")
        
        # Create test data
        test_data = torch.randn(200, 3, 32, 32)
        test_targets = torch.randint(0, 10, (200,))
        test_loader = DataLoader(
            TensorDataset(test_data, test_targets),
            batch_size=32, shuffle=True
        )
        
        results = {}
        
        for attack_type in attack_types:
            results[attack_type] = {}
            
            for epochs in epochs_list:
                print(f"  Testing {attack_type} with {epochs} epochs...")
                
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                
                if attack_type == 'dina':
                    # Create simple teacher model
                    teacher = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(3 * 32 * 32, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)
                    ).to(self.device)
                    
                    attacker = DINAAttacker(
                        input_shape=(3, 32, 32),
                        num_classes=10,
                        device=self.device
                    )
                    
                    try:
                        history = attacker.train_distillation(
                            teacher_model=teacher,
                            data_loader=test_loader,
                            epochs=epochs,
                            lr=1e-3
                        )
                    except Exception as e:
                        print(f"    Error: {e}")
                        continue
                
                elif attack_type == 'mla':
                    # Create dummy features
                    features = torch.randn(200, 128)
                    feature_loader = DataLoader(
                        TensorDataset(features, test_data),
                        batch_size=32, shuffle=True
                    )
                    
                    attacker = MLAAttacker(
                        input_shape=(3, 32, 32),
                        feature_shape=(128,),
                        device=self.device
                    )
                    
                    try:
                        history = attacker.train_attack(
                            feature_loader=feature_loader,
                            target_loader=feature_loader,
                            epochs=epochs,
                            lr=1e-3
                        )
                    except Exception as e:
                        print(f"    Error: {e}")
                        continue
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                
                results[attack_type][epochs] = {
                    'training_time': end_time - start_time,
                    'memory_usage_mb': end_memory - start_memory,
                    'time_per_epoch': (end_time - start_time) / epochs
                }
                
                print(f"    {epochs} epochs: {end_time - start_time:.3f}s, "
                      f"{(end_time - start_time) / epochs:.3f}s/epoch")
        
        self.results['attack_training'] = results
        return results
    
    def benchmark_boundary_finding(self, model_name='vgg11', max_layers=3):
        """Benchmark complete boundary finding process."""
        print("Benchmarking boundary finding...")
        
        # Load model and data
        model = get_model(model_name, num_classes=10, pretrained=False)
        model = model.to(self.device)
        
        # Create small dataset for speed
        test_data = torch.randn(100, 3, 32, 32)
        test_targets = torch.randint(0, 10, (100,))
        test_loader = DataLoader(
            TensorDataset(test_data, test_targets),
            batch_size=32, shuffle=False
        )
        
        # Get limited layer candidates
        all_candidates = identify_layer_candidates(model)
        candidates = all_candidates[:max_layers]
        
        config = C2PIConfig(
            privacy_threshold=0.3,
            accuracy_threshold=0.95,
            device=self.device
        )
        
        boundary_finder = BoundaryFinder(
            model=model,
            layer_candidates=candidates,
            config=config
        )
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            # Run with minimal epochs for speed
            results = boundary_finder.find_optimal_boundary(
                train_loader=test_loader,
                test_loader=test_loader,
                attack_type='dina',
                attack_epochs=3,  # Very few epochs for benchmarking
                attack_lr=1e-3
            )
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            benchmark_results = {
                'total_time': end_time - start_time,
                'memory_usage_mb': end_memory - start_memory,
                'layers_evaluated': len(candidates),
                'time_per_layer': (end_time - start_time) / len(candidates),
                'successful': True
            }
            
            print(f"  Total time: {end_time - start_time:.3f}s")
            print(f"  Time per layer: {(end_time - start_time) / len(candidates):.3f}s")
            print(f"  Memory usage: {end_memory - start_memory:.1f}MB")
            
        except Exception as e:
            print(f"  Error in boundary finding: {e}")
            benchmark_results = {
                'total_time': time.time() - start_time,
                'successful': False,
                'error': str(e)
            }
        
        self.results['boundary_finding'] = benchmark_results
        return benchmark_results
    
    def run_all_benchmarks(self):
        """Run all benchmarks and return comprehensive results."""
        print("Running C2PI Performance Benchmarks")
        print("=" * 50)
        
        # System info
        print(f"Device: {self.device}")
        print(f"CPU: {psutil.cpu_count()} cores")
        print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print()
        
        # Run individual benchmarks
        self.benchmark_model_loading()
        print()
        
        self.benchmark_layer_identification()
        print()
        
        self.benchmark_feature_extraction()
        print()
        
        self.benchmark_attack_training()
        print()
        
        self.benchmark_boundary_finding()
        print()
        
        return self.results
    
    def save_results(self, filename='c2pi_benchmark_results.json'):
        """Save benchmark results to file."""
        import json
        
        # Convert any torch tensors to lists
        def convert_tensors(obj):
            if torch.is_tensor(obj):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        results_json = convert_tensors(self.results)
        
        with open(filename, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Benchmark results saved to: {filename}")


def main():
    """Run performance benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='C2PI Performance Benchmarks')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--output', type=str, default='c2pi_benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmarks only')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Run benchmarks
    benchmark = PerformanceBenchmark(device)
    
    if args.quick:
        # Quick benchmarks only
        benchmark.benchmark_model_loading(['vgg11'])
        benchmark.benchmark_layer_identification(['vgg11'])
        benchmark.benchmark_feature_extraction(data_sizes=[100])
        benchmark.benchmark_attack_training(epochs_list=[5])
    else:
        # Full benchmark suite
        benchmark.run_all_benchmarks()
    
    # Save results
    benchmark.save_results(args.output)
    
    print("Benchmarking completed!")


if __name__ == '__main__':
    main()
