import time

#!/usr/bin/env python3
import launcher as launcher
import argparse
import os
import importlib
import torch
# Assume the launcher exposes a list of available models as MODELS.
# If the list name differs, adjust accordingly.
models = ['TwoLayerCNN', 'SixLayerCNN', 'MobileNetV2', 'ResNet18']

def fprint(msg):
    print(msg)
    with open(f"results/exp3.txt", "a") as f:
        f.write(msg + "\n")

for model_name in models:
    total_times = []
    for i in range(5):
        #Create the model and save it in data directory
        model_module = importlib.import_module("model")
        ModelClass = getattr(model_module, model_name)
        model = ModelClass()
        torch.save(model.state_dict(), f"data/model.pth")
        start = time.time()
        os.system(f"python3 launcher.py --world_size 2 --seed 42 --model_name {model_name}")
        elapsed = time.time() - start
        total_times.append(elapsed)
    # model_name = getattr(model, "__name__", str(model))
    avg_time = sum(total_times) / len(total_times)
    fprint(f"Model {model_name} executed 5 times in {avg_time:.3f} seconds")
    # print(f"Model {model_name} executed 5 times in {elapsed:.2f} seconds")
    