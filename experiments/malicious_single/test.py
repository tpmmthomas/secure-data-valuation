import os
import importlib
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import ezkl
import os
import torchvision
import json
import random 
import numpy as np
import torchvision.transforms as transforms
from launcher import run_experiment


#Specifying some path parameters
model_path = os.path.join('data','network.onnx')
compiled_model_path = os.path.join('data','network.compiled')
pk_path = os.path.join('data','test.pk')
vk_path = os.path.join('data','test.vk')
settings_path = os.path.join('data','settings.json')
cal_path = os.path.join('data',"calibration.json")
witness_path = os.path.join('data','witness.json')
data_path = os.path.join('data','input.json')
output_path = os.path.join('data','output.json')
label_path = os.path.join('data','label.json')
proof_path = os.path.join('data','test.pf')

# List of models to benchmark.
models = ["ResNet18"] # 

with open("results/exp4.txt", "w") as f:
        pass
    
def fprint(msg,fn="results/exp4.txt"):
    print(msg)
    with open(fn, "a") as f:
        f.write(msg + "\n")
        
# Set torch seeds
torch.manual_seed(42)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True,transform=transform)
indices = random.sample(range(len(trainset)), 3)
cal_images = np.array([trainset[i][0].numpy() for i in indices])
data_array = (cal_images).reshape([-1]).tolist()

data = dict(input_data = [data_array])

# Serialize data into file:
json.dump(data, open(cal_path, 'w'))
async def main():
    for model_name in models:
        total_times = []
        total_memory = []
        total_precision = []
        total_comm = []
        # Create the model and save it in the data directory.
        model_module = importlib.import_module("model")
        ModelClass = getattr(model_module, model_name)
        model = ModelClass()
        torch.save(model.state_dict(), "data/model.pth")
        #Setup the model
        data = torch.load("data/data.pth")
        torch.onnx.export(model,               # model being run
                        data,                   # model input (or a tuple for multiple inputs)
                        model_path,            # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=10,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
        
        py_run_args = ezkl.PyRunArgs()
        py_run_args.input_visibility = "public" #Bob can see this
        py_run_args.output_visibility = "hashed/public" #This hash is given to Bob
        py_run_args.param_visibility = "private" 
        py_run_args.logrows = 14
        print("Generating settings")
        res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
        assert res
        # await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")
        print("Compiling circuit")
        res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
        assert res
        # print("Generating srs")
        res = await ezkl.get_srs(settings_path)
        print("Setup here")
        res = ezkl.setup(
            compiled_model_path,
            vk_path,
            pk_path,
        )
        print("Done!")

import asyncio
asyncio.run(main())