#!/bin/bash
# Basic training script for CIFAR-10 knowledge distillation

echo "Starting CIFAR-10 Knowledge Distillation Training..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export OMP_NUM_THREADS=4

# Training parameters
CONFIG_FILE="configs/experiments/cifar10_resnet18_cnn5.yaml"
OUTPUT_DIR="artifacts/$(date +%Y%m%d_%H%M%S)_cifar10_baseline"

# Run training
python src/main.py \
    --config $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --batch_size 128 \
    --epochs_teacher 5 \
    --epochs_student 20 \
    --alpha 0.5 \
    --temperature 4.0

echo "Training completed. Results saved to: $OUTPUT_DIR"

# Generate summary report
echo "Generating summary report..."
python scripts/generate_report.py --experiment_dir $OUTPUT_DIR

echo "Done!"
