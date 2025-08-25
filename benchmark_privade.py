#!/usr/bin/env python3
"""
PrivaDE Benchmarking Script

This script benchmarks all major components of the PrivaDE workflow:
- Model training and distillation
- Model splitting
- Weight mixing
- Representative set selection (dimension reduction + clustering)
- Challenge protocol
- Model inference

Results are saved to a CSV file with timing and performance metrics.
"""

import os
import sys
import time
import random
import csv
import traceback
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add privade directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'privade'))

from privade.data import get_dataset
from privade.models import get_model
from privade.distillation import train_distilled_model
from privade.split import split_model
from privade.weight_mixer import weight_mixer
from privade.dim_reduction import reduce_image_dimensions
from privade.clustering import kmeans_clustering
from privade.challenge_protocol import create_proof, verify_proof


class PrivadeBenchmark:
    """Benchmarking class for PrivaDE components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': config.copy()
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create data directory if it doesn't exist
        os.makedirs('benchmark_data', exist_ok=True)
        
    def benchmark_data_preparation(self) -> Dict[str, Any]:
        """Benchmark data loading and preparation."""
        print("\n=== Benchmarking Data Preparation ===")
        start_time = time.time()
        
        try:
            # Load full dataset
            full_data = get_dataset(self.config['dataset'])
            
            # Randomly select data for Alice and Bob
            N = self.config['dataset_size']
            indices_bob = random.sample(range(len(full_data)), N)
            bob_images = np.array([full_data[i][0].numpy() for i in indices_bob])
            bob_labels = np.array([full_data[i][1] for i in indices_bob])
            
            indices_alice = random.sample(range(len(full_data)), N)
            alice_images = np.array([full_data[i][0].numpy() for i in indices_alice])
            alice_labels = np.array([full_data[i][1] for i in indices_alice])
            
            # Create tensors and dataloaders
            alice_images_tensor = torch.FloatTensor(alice_images)
            alice_labels_tensor = torch.LongTensor(alice_labels)
            alice_dataset = TensorDataset(alice_images_tensor, alice_labels_tensor)
            alice_dataloader = DataLoader(alice_dataset, batch_size=self.config['batch_size'], shuffle=True)
            
            end_time = time.time()
            
            # Store data for later use
            self.bob_images = bob_images
            self.bob_labels = bob_labels
            self.alice_images = alice_images
            self.alice_labels = alice_labels
            self.alice_dataloader = alice_dataloader
            self.alice_dataset = alice_dataset
            
            return {
                'component': 'data_preparation',
                'time_seconds': end_time - start_time,
                'dataset_size': N,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'component': 'data_preparation',
                'time_seconds': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def benchmark_model_training(self) -> Dict[str, Any]:
        """Benchmark initial model training."""
        print("\n=== Benchmarking Model Training ===")
        start_time = time.time()
        
        try:
            # Create and train full model
            full_model = get_model(self.config['model_name'], self.config['dataset'])
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(full_model.parameters(), lr=self.config['learning_rate'])
            
            full_model.train()
            total_loss = 0
            
            for epoch in range(self.config['training_epochs']):
                epoch_loss = 0
                for batch_idx, (images, labels) in enumerate(self.alice_dataloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = full_model(images.to(self.device))
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                total_loss += epoch_loss
                print(f"Epoch {epoch+1}/{self.config['training_epochs']}, Loss: {epoch_loss/len(self.alice_dataloader):.4f}")
            
            end_time = time.time()
            
            # Store model for later use
            self.full_model = full_model
            
            return {
                'component': 'model_training',
                'time_seconds': end_time - start_time,
                'epochs': self.config['training_epochs'],
                'final_loss': total_loss / (self.config['training_epochs'] * len(self.alice_dataloader)),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'component': 'model_training',
                'time_seconds': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def benchmark_model_distillation(self) -> Dict[str, Any]:
        """Benchmark model distillation."""
        print("\n=== Benchmarking Model Distillation ===")
        start_time = time.time()
        
        try:
            # Create student model
            student_model = get_model(self.config['student_model'], self.config['dataset'])
            
            # Train distilled model
            kd_train_loader = DataLoader(self.alice_dataset, batch_size=self.config['batch_size'], shuffle=True)
            trained_student_model = train_distilled_model(
                self.full_model, 
                student_model, 
                kd_train_loader, 
                kd_train_loader,
                epochs=self.config['distillation_epochs']
            )
            
            end_time = time.time()
            
            # Store model for later use
            self.trained_student_model = trained_student_model
            
            return {
                'component': 'model_distillation',
                'time_seconds': end_time - start_time,
                'epochs': self.config['distillation_epochs'],
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'component': 'model_distillation',
                'time_seconds': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def benchmark_model_splitting(self) -> Dict[str, Any]:
        """Benchmark model splitting."""
        print("\n=== Benchmarking Model Splitting ===")
        start_time = time.time()
        
        try:
            # Split the model
            model_A, model_B, model_C, split_stats = split_model(
                data_loader=self.alice_dataloader,
                model=self.trained_student_model,
            )
            
            end_time = time.time()
            
            # Store models for later use
            self.model_A = model_A.to(self.device)
            self.model_B = model_B.to(self.device)
            self.model_C = model_C.to(self.device)
            self.split_stats = split_stats
            
            return {
                'component': 'model_splitting',
                'time_seconds': end_time - start_time,
                'optimal_layer': split_stats['optimal_layer'],
                'privacy_preserved_rate': split_stats['privacy_preserved_rate'],
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'component': 'model_splitting',
                'time_seconds': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def benchmark_weight_mixing(self) -> Dict[str, Any]:
        """Benchmark weight mixing."""
        print("\n=== Benchmarking Weight Mixing ===")
        start_time = time.time()
        
        try:
            # Apply weight mixing
            self.model_A, self.model_B = weight_mixer(self.model_A, self.model_B)
            
            end_time = time.time()
            
            return {
                'component': 'weight_mixing',
                'time_seconds': end_time - start_time,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'component': 'weight_mixing',
                'time_seconds': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def benchmark_dimension_reduction(self) -> Dict[str, Any]:
        """Benchmark dimension reduction."""
        print("\n=== Benchmarking Dimension Reduction ===")
        start_time = time.time()
        
        try:
            target_dimension = self.config['target_dimension']
            reduced_images, _, _ = reduce_image_dimensions(self.bob_images, target_dimension)
            
            end_time = time.time()
            
            # Store for later use
            self.reduced_images = reduced_images
            
            return {
                'component': 'dimension_reduction',
                'time_seconds': end_time - start_time,
                'original_dimension': np.prod(self.bob_images[0].shape),
                'target_dimension': target_dimension,
                'compression_ratio': np.prod(self.bob_images[0].shape) / target_dimension,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'component': 'dimension_reduction',
                'time_seconds': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def benchmark_clustering(self) -> Dict[str, Any]:
        """Benchmark clustering for representative set selection."""
        print("\n=== Benchmarking Clustering ===")
        start_time = time.time()
        
        try:
            rep_set_size = self.config['rep_set_size']
            representative_set = kmeans_clustering(self.reduced_images, rep_set_size)
            
            end_time = time.time()
            
            # Store for later use
            self.representative_set = representative_set
            self.representative_points = self.reduced_images[representative_set]
            self.rep_points = self.bob_images[representative_set]
            self.rep_labels = self.bob_labels[representative_set]
            
            # Calculate distances for challenge protocol
            dists = np.linalg.norm(self.reduced_images[:, None] - self.representative_points[None, :], axis=2)
            min_dists = np.min(dists, axis=1)
            self.max_min_distance = np.ceil(np.max(min_dists))
            
            return {
                'component': 'clustering',
                'time_seconds': end_time - start_time,
                'rep_set_size': rep_set_size,
                'max_min_distance': float(self.max_min_distance),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'component': 'clustering',
                'time_seconds': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def benchmark_challenge_protocol(self) -> Dict[str, Any]:
        """Benchmark challenge protocol."""
        print("\n=== Benchmarking Challenge Protocol ===")
        start_time = time.time()
        
        try:
            M = self.config['challenge_number']
            N = len(self.bob_images)
            
            # Alice randomly selects M points from the whole dataset
            indices = random.sample(range(N), M)
            
            total_proof_time = 0
            total_verify_time = 0
            successful_proofs = 0
            
            for idx in indices:
                # Find the index from representative_points which has the min distance
                selected_point = self.reduced_images[idx]
                dists = np.linalg.norm(self.representative_points - selected_point, axis=1)
                min_index = np.argmin(dists)
                
                cp_data = {
                    "messageArray": selected_point.tolist(),
                    "idx": int(min_index),
                    "allPoints": self.representative_points.tolist(),
                    "d": int(self.max_min_distance),
                    "r": 0x12345678
                }
                
                proof_file = f"benchmark_data/proof_{idx}.json"
                
                # Time proof creation
                proof_start = time.time()
                proof_success = create_proof(cp_data, proof_file)
                proof_time = time.time() - proof_start
                total_proof_time += proof_time
                
                if proof_success:
                    # Time proof verification
                    verify_start = time.time()
                    verify_success = verify_proof(proof_file)
                    verify_time = time.time() - verify_start
                    total_verify_time += verify_time
                    
                    if verify_success:
                        successful_proofs += 1
            
            end_time = time.time()
            
            return {
                'component': 'challenge_protocol',
                'time_seconds': end_time - start_time,
                'challenge_number': M,
                'successful_proofs': successful_proofs,
                'success_rate': successful_proofs / M,
                'avg_proof_time': total_proof_time / M,
                'avg_verify_time': total_verify_time / M,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'component': 'challenge_protocol',
                'time_seconds': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def benchmark_model_inference(self) -> Dict[str, Any]:
        """Benchmark model inference for all three parts."""
        print("\n=== Benchmarking Model Inference ===")
        start_time = time.time()
        
        try:
            # Model A inference (2PC)
            model_a_start = time.time()
            self.model_A = self.model_A.to(self.device)
            y_a = self.model_A(torch.tensor(self.rep_points, dtype=torch.float32).to(self.device))
            model_a_time = time.time() - model_a_start
            
            # Model B inference (Bob's side)
            model_b_start = time.time()
            self.model_B = self.model_B.to(self.device)
            y_b = self.model_B(y_a)
            model_b_time = time.time() - model_b_start
            
            # Model C inference (Alice's side)
            model_c_start = time.time()
            self.model_C = self.model_C.to(self.device)
            y_c = self.model_C(y_b)
            model_c_time = time.time() - model_c_start
            
            end_time = time.time()
            
            return {
                'component': 'model_inference',
                'time_seconds': end_time - start_time,
                'model_a_time': model_a_time,
                'model_b_time': model_b_time,
                'model_c_time': model_c_time,
                'output_shape': list(y_c.shape),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'component': 'model_inference',
                'time_seconds': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("Starting PrivaDE Benchmark...")
        print(f"Configuration: {self.config}")
        
        benchmark_start = time.time()
        
        # Run all benchmark components
        components = [
            self.benchmark_data_preparation,
            self.benchmark_model_training,
            self.benchmark_model_distillation,
            self.benchmark_model_splitting,
            self.benchmark_weight_mixing,
            self.benchmark_dimension_reduction,
            self.benchmark_clustering,
            self.benchmark_challenge_protocol,
            self.benchmark_model_inference
        ]
        
        for component_func in components:
            try:
                result = component_func()
                self.results[result['component']] = result
                
                if result['success']:
                    print(f"✓ {result['component']}: {result['time_seconds']:.3f}s")
                else:
                    print(f"✗ {result['component']}: FAILED - {result['error']}")
                    
            except Exception as e:
                component_name = component_func.__name__.replace('benchmark_', '')
                print(f"✗ {component_name}: CRASHED - {str(e)}")
                traceback.print_exc()
                
                self.results[component_name] = {
                    'component': component_name,
                    'time_seconds': 0,
                    'success': False,
                    'error': str(e)
                }
        
        total_time = time.time() - benchmark_start
        self.results['total_time'] = total_time
        
        print(f"\nBenchmark completed in {total_time:.3f}s")
        return self.results
    
    def save_results_to_csv(self, filename: str):
        """Save benchmark results to CSV file."""
        print(f"\nSaving results to {filename}...")
        
        # Flatten the results for CSV
        csv_data = []
        for component, data in self.results.items():
            if component in ['timestamp', 'config', 'total_time']:
                continue
                
            if isinstance(data, dict):
                row = {
                    'timestamp': self.results['timestamp'],
                    'component': component,
                    **data,
                    'total_benchmark_time': self.results['total_time']
                }
                # Add config parameters
                for key, value in self.results['config'].items():
                    row[f'config_{key}'] = value
                    
                csv_data.append(row)
        
        # Write to CSV
        if csv_data:
            fieldnames = csv_data[0].keys()
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"Results saved to {filename}")
        else:
            print("No data to save!")


def main():
    """Main benchmarking function."""
    
    # Default configuration
    config = {
        'dataset': 'mnist',
        'model_name': 'lenet5',
        'student_model': 'lenetxs',
        'dataset_size': 1000,
        'batch_size': 32,
        'learning_rate': 0.001,
        'training_epochs': 5,
        'distillation_epochs': 10,
        'target_dimension': 50,
        'rep_set_size': 20,
        'challenge_number': 20
    }
    
    # Create benchmark instance
    benchmark = PrivadeBenchmark(config)
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"privade_benchmark_{timestamp}.csv"
    benchmark.save_results_to_csv(csv_filename)
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    successful_components = 0
    total_components = 0
    
    for component, data in results.items():
        if component in ['timestamp', 'config', 'total_time']:
            continue
            
        total_components += 1
        if isinstance(data, dict) and data.get('success', False):
            successful_components += 1
            print(f"✓ {component:<25} {data['time_seconds']:>8.3f}s")
        else:
            error_msg = data.get('error', 'Unknown error') if isinstance(data, dict) else str(data)
            print(f"✗ {component:<25} FAILED ({error_msg})")
    
    print("-" * 50)
    print(f"Success Rate: {successful_components}/{total_components} ({100*successful_components/total_components:.1f}%)")
    print(f"Total Time: {results['total_time']:.3f}s")
    print(f"Results saved to: {csv_filename}")


if __name__ == "__main__":
    main()
