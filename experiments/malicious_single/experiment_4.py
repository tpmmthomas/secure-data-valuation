import time
import os
import importlib
import torch
import subprocess
import re
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
import asyncio

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
models = ['MobileNetV2'] #'SVM', 'LeNet', 'AlexNet',

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
        py_run_args.logrows = 16 #log rows = 14 used to work for ResNet18
        print("Generating settings")
        res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
        assert res
        # await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")
        print("Compiling circuit")
        res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
        assert res
        print("Generating srs")
        res = await ezkl.get_srs(settings_path)
        print("Setup here")
        res = ezkl.setup(
            compiled_model_path,
            vk_path,
            pk_path,
        )
        data_array = ((data).detach().numpy()).reshape([-1]).tolist()
        data = dict(input_data = [data_array])
            # Serialize data into file:
        json.dump( data, open(data_path, 'w' ))
        #Prep input for MPC
        model_module = importlib.import_module("model")
        ModelClass = getattr(model_module, model_name)
        model = ModelClass()
        data = torch.load("data/data.pth")
        label = torch.load("data/lbl.pth")
        model.eval()
        output = model(data)
        mpcout = F.softmax(output, dim=1)
        mpc_pred = mpcout.detach().numpy()[0]
        mpc_gt = label.detach().numpy()
        #
        if not os.path.exists('../../MP-SPDZ/Player-Data'):
            os.makedirs('../../MP-SPDZ/Player-Data')
        p0_path = os.path.join('../../MP-SPDZ/Player-Data','Input-P0-0')
        p1_path = os.path.join('../../MP-SPDZ/Player-Data','Input-P1-0')
        with open(p0_path, 'w') as f:
            for i in mpc_gt:
                f.write(f'{i:.6f} ')
            f.write('\n')
        with open(p1_path, 'w') as f:
            for i in mpc_pred:
                f.write(f'{i:.6f} ')
            f.write('\n')
        #Repeat experiment 5 times
        for i in range(5):
            print("Running experiment", i)
            data = torch.load("data/data.pth")
            label = torch.load("data/lbl.pth")
            loss = nn.CrossEntropyLoss()
            model.eval()
            output = model(data)
            _, label = label.max(dim=0)
            label = label.unsqueeze(0)
            loss_value = loss(output, label).item()
            # Build the command. The '/usr/bin/time -v' prefix will output detailed resource usage.
            command = f"/usr/bin/time -v python launcher.py"
            
            start = time.time()
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            elapsed = time.time() - start
            total_times.append(elapsed)
            for line in result.stdout.splitlines():
                fprint(line,f'results/online_offline_{model_name}.txt')
        #     # Parse the memory usage from result.stderr.
        #     # Look for a line like: "Maximum resident set size (kbytes): 12345"
            mem_usage = None
            for line in result.stderr.splitlines():
                if "Maximum resident set size" in line:
                    try:
                        mem_usage = int(line.split(":")[-1].strip())
                    except ValueError:
                        mem_usage = None
                    break
            
            if mem_usage is not None:
                total_memory.append(mem_usage)
            else:
                print("Warning: Memory usage not found in output.")
            
            # Calculate precision
            for line in result.stdout.splitlines():
                if "cross entropy" in line.lower():
                    secure_loss_value = line.split(" ")[-1].strip()
                    secure_loss_value = float(secure_loss_value)
                if "global data" in line.lower():
                    data_sent = line.split(" ")[4].strip()
                    data_sent = float(data_sent)
                    data_file_size = os.path.getsize("data/data.pth") / (1024 * 1024)
                    data_sent += data_file_size
            
            # print(secure_loss_value, loss_value)
            diff = secure_loss_value - loss_value
            percentage_diff = (diff / loss_value) * 100
            total_precision.append(percentage_diff)
            
            #Get proof size in MB
            proof_size = os.path.getsize(proof_path) / (1024 * 1024)
            total_comm.append(proof_size + data_sent)

        
        avg_time = sum(total_times) / len(total_times)
        avg_memory = sum(total_memory) / len(total_memory) if total_memory else 0
        avg_precision = sum(total_precision) / len(total_precision)
        avg_comm = sum(total_comm) / len(total_comm) if total_comm else 0
        fprint(f"Model {model_name} executed 5 times, average time {avg_time:.3f} seconds with average memory usage {avg_memory:.1f} kB and average precision diff {avg_precision:.3f}%, avg comm overhead {avg_comm:.3f} MB")
        #Export all time, memory and precision to csv
        results_file = f"results/exp4_{model_name}.txt"
        fprint(f"times: {total_times}", results_file)
        fprint(f"memory: {total_memory}", results_file)
        fprint(f"precision: {total_precision}", results_file)
        fprint(f"comm: {total_comm}", results_file)


asyncio.run(main())