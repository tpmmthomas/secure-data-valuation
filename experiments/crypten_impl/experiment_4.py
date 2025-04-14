import time
import os
import importlib
import torch
import subprocess
import re
import torch.nn as nn
import pandas as pd

# List of models to benchmark.
models = ['TwoLayerCNN', 'SixLayerCNN', 'MobileNetV2', 'ResNet18']

with open(f"results/exp4.txt", "w") as f:
        pass
    
def fprint(msg,fn="results/exp4.txt"):
    print(msg)
    with open(fn, "a") as f:
        f.write(msg + "\n")
        
# Set torch seeds
torch.manual_seed(42)

for model_name in models:
    total_times = []
    total_memory = []
    total_precision = []
    total_comm = []
    for i in range(5):
        # Create the model and save it in the data directory.
        model_module = importlib.import_module("model")
        ModelClass = getattr(model_module, model_name)
        model = ModelClass()
        torch.save(model.state_dict(), "data/model.pth")
        
        data = torch.load("data/data.pth")
        label = torch.load("data/lbl.pth")
        loss = nn.CrossEntropyLoss()
        model.eval()
        output = model(data)
        _, label = label.max(dim=0)
        label = label.unsqueeze(0)
        loss_value = loss(output, label).item()
        # Build the command. The '/usr/bin/time -v' prefix will output detailed resource usage.
        command = f"/usr/bin/time -v python launcher.py --world_size 2 --seed 42 --model_name {model_name}"
        
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


        # Parse the memory usage from result.stderr.
        # Look for a line like: "Maximum resident set size (kbytes): 12345"
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
        # with open("results/data_valuation.txt", "r") as f:
        #     secure_loss_value = float(f.read())
        secure_loss_value = None
        for line in result.stdout.splitlines():
            if "Loss value" in line:
                secure_loss_value = line.split(":")[-1].strip()
                match = re.search(r"tensor\((.*?)\)", secure_loss_value)
                if match:
                    secure_loss_value = float(match.group(1)) * 10 
            if "Communication" in line:
                comm = float(line.split(' ')[6][:-1])
        if not secure_loss_value:
            for line in result.stderr.splitlines():
                print(line)
        # print(secure_loss_value, loss_value)
        diff = secure_loss_value - loss_value
        percentage_diff = (diff / loss_value) * 100
        total_precision.append(percentage_diff)
        total_comm.append(comm)
    
    avg_time = sum(total_times) / len(total_times)
    avg_memory = sum(total_memory) / len(total_memory) if total_memory else 0
    avg_precision = sum(total_precision) / len(total_precision)
    avg_comm = sum(total_comm) / len(total_comm)
    # avg_precision = 0
    fprint(f"Model {model_name} executed 5 times, average time {avg_time:.3f} seconds with average memory usage {avg_memory:.1f} kB and average precision diff {avg_precision:.3f}%, average communication {avg_comm:.3f} Bytes")
    #Export all time, memory and precision to csv
    results_file = f"results/exp4_{model_name}.txt"
    fprint(f"times: {total_times}", results_file)
    fprint(f"memory: {total_memory}", results_file)
    fprint(f"precision: {total_precision}", results_file)
    fprint(f"comm: {total_comm}", results_file)
    # row = {"Model": model_name, "Avg_Time": avg_time, "Avg_Memory": avg_memory, "Avg_Precision": avg_precision}
    # df_row = pd.DataFrame([row])
    # if not os.path.exists(results_file):
    #     df_row.to_csv(results_file, index=False)
    # else:
    #     df_row.to_csv(results_file, mode="a", index=False, header=False)
    # fprint(f"Model {model_name} executed 5 times, average time {avg_time:.3f} seconds with average memory usage {avg_memory:.1f} kB")
