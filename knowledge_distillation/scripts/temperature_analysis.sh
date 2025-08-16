#!/bin/bash
# Temperature analysis script

echo "Running temperature analysis for knowledge distillation..."

# Base configuration
BASE_CONFIG="configs/experiments/temperature_analysis.yaml"
BASE_OUTPUT="artifacts/temperature_analysis"

# Temperature values to test
TEMPERATURES=(1.0 2.0 4.0 6.0 8.0 12.0 16.0)

for temp in "${TEMPERATURES[@]}"; do
    echo "Training with temperature: $temp"
    
    OUTPUT_DIR="${BASE_OUTPUT}/temp_${temp}"
    
    python src/main.py \
        --config $BASE_CONFIG \
        --output_dir $OUTPUT_DIR \
        --temperature $temp \
        --epochs_teacher 3 \
        --epochs_student 15
    
    echo "Completed temperature $temp"
done

# Generate comparison report
echo "Generating temperature analysis report..."
python scripts/analyze_temperature.py --base_dir $BASE_OUTPUT

echo "Temperature analysis complete!"
