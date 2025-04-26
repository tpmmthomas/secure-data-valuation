#!/usr/bin/env python3
import sys
import onnx
import pathlib

def count_onnx_parameters(onnx_path: str) -> int:
    """
    Load an ONNX model and count the total number of parameters
    (sum of all elements in each initializer tensor).
    """
    model = onnx.load(onnx_path)
    total_params = 0
    for tensor in model.graph.initializer:
        # tensor.dims is a tuple of the shape, e.g. (64, 3, 7, 7)
        num_elements = 1
        for dim in tensor.dims:
            num_elements *= dim
        total_params += num_elements
    return total_params

data = pathlib.Path('./data')

for models in data.glob('AlexNet*.onnx'):
    model_path = models.resolve()
    print(f"Model: {model_path}")
    total_params = count_onnx_parameters(str(model_path))
    print(f"Total parameters: {total_params}")
    print(f"Total size (MB): {total_params * 4 / (1024 * 1024)}")
    print()