import torch
import torch.nn as nn
import torch.nn.functional as F
import ezkl
import os
import json
import importlib
import argparse
import time

#Specifying some path parameters
model_path = os.path.join('data','network.onnx')
compiled_model_path = os.path.join('data','network.compiled')
pk_path = os.path.join('data','test.pk')
vk_path = os.path.join('data','test.vk')
settings_path = os.path.join('data','settings.json')

witness_path = os.path.join('data','witness.json')
data_path = os.path.join('data','input.json')
output_path = os.path.join('data','output.json')
label_path = os.path.join('data','label.json')
proof_path = os.path.join('data','test.pf')

parser = argparse.ArgumentParser(description="CrypTen Cifar Training")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--model_name",
    type=str,
    default="Model",
    help="Specifies the model name",
)

async def run_experiment():
    #Simulate ZKP component
    start = time.time()
    res = await ezkl.gen_witness(data_path, compiled_model_path, witness_path) #Put this later
    res = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "single",
    )
    end = time.time()
    print(f"ZKP Proof Time: {end-start}")
    start = time.time()
    res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
    )
    os.system("cd ../../MP-SPDZ/ && Scripts/spdz2k.sh dataval_ce")
    end = time.time()
    print(f"ZKP Online Time: {end-start}")
    

import asyncio
if __name__ == "__main__":
    asyncio.run(run_experiment())