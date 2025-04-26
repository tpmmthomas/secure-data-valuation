import torch
import torch.nn as nn
import torch.nn.functional as F
import ezkl
import os
import json
import importlib
import argparse
import random
import time
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


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

image_path = "data/selected_images.pth"
labels_path = "data/selected_labels.pth"

N = 1000 #Bob's dataset size
DIM = 50 # Dimension of the reduced dataset
K = 20 # Representative set size
M = 20 # Number of CP checks by Alice


async def run_experiment(model_name):
    #Simulate Bob loading data
    selected_images = torch.load(image_path).numpy()
    selected_labels = torch.load(labels_path).numpy()
    model_module = importlib.import_module("model")
    ModelClass = getattr(model_module, model_name)
    model = ModelClass()
    model.eval()
    #Dimension reduction
    n_components = DIM 
    random_projection = SparseRandomProjection(n_components=n_components, random_state=42)
    flattened_images = selected_images.reshape(selected_images.shape[0], -1)
    reduced_images = random_projection.fit_transform(flattened_images)
    scaler = MinMaxScaler(feature_range=(0, 1))
    reduced_images = scaler.fit_transform(reduced_images)
    torch.save(torch.tensor(reduced_images), 'data/reduced_images.pth')
    torch.save(torch.tensor(selected_labels), 'data/reduced_labels.pth')
    #Clustering
    print("Clustering...")
    flattened_images = reduced_images.reshape(reduced_images.shape[0], -1)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(flattened_images)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    representative_set = []
    for i in range(K):
        candidate_indices = np.where(cluster_labels == i)[0]
        distances = np.linalg.norm(flattened_images[candidate_indices] - cluster_centers[i], axis=1)
        representative_set.append(candidate_indices[np.argmin(distances)])
    representative_set = np.array(representative_set)
    #Max dist for challenge protocol
    representative_points = reduced_images[representative_set]
    dists = np.linalg.norm(reduced_images[:, None] - representative_points[None, :], axis=2)
    min_dists = np.min(dists, axis=1)
    max_min_distance = np.ceil(np.max(min_dists))
    #Challenge Protocol
    print("Challenge Protocol...")
    indices = random.sample(range(N), M)
    for idx in indices:
        print("Running challenge protocol for index:", idx)
        selected_point = reduced_images[idx]
        dists = np.linalg.norm(representative_points - selected_point, axis=1)
        min_index = np.argmin(dists)
        print(np.min(dists), min_index)
        cp_data = {
            "messageArray": selected_point.tolist(),
            "idx": int(min_index),
            "allPoints": representative_points.tolist(),
            "d": int(max_min_distance),
            "r": 0x12345678
        }
        assert len(selected_point.tolist()) == 50
        assert len(representative_points.tolist()) == 20
        with open('data/cp.json', 'w') as f:
            json.dump(cp_data, f)
        exit_code = os.system("node commit.js")
        assert exit_code == 0, "Command 'node commit.js' failed"
        # Generate the witness
        exit_code = os.system("node cp_overall_js/generate_witness.js cp_overall_js/cp_overall.wasm input.json witness.wtns")
        assert exit_code == 0, "Command to generate witness failed"
        # Generate the proof
        exit_code = os.system("snarkjs groth16 prove cp_0001.zkey witness.wtns proof.json public.json")
        assert exit_code == 0, "Command to generate proof failed"
        # Verify the proof
        exit_code = os.system("snarkjs groth16 verify verification_key.json public.json proof.json")
        assert exit_code == 0, "Command to verify proof failed"
    #ZKP for model inferences
    print("ZKP for model inferences...")
    points_to_submit = selected_images[representative_set]
    labels_to_submit = selected_labels[representative_set]
    outputs = []
    inp = torch.tensor(points_to_submit)
    output = model(inp)
    output = output.detach().numpy()
    outputs.append(output)
    data_array = [img.reshape(-1).tolist() for img in points_to_submit]
    data = dict(input_data = data_array)
    json.dump(data, open(data_path, 'w'))
    res = await ezkl.gen_witness(data_path, compiled_model_path, witness_path) #Put this later
    assert res
    res = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "single",
    )
    assert res
    res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
    )
    assert res
    #Prepare data for MPC
    points_to_submit = reduced_images[representative_set]
    if not os.path.exists('../../MP-SPDZ/Player-Data'):
        os.makedirs('../../MP-SPDZ/Player-Data')
    p0_path = os.path.join('../../MP-SPDZ/Player-Data','Input-P0-0')
    p1_path = os.path.join('../../MP-SPDZ/Player-Data','Input-P1-0')
    points_1d = np.array(points_to_submit).reshape(-1).tolist()
    one_hot_labels = np.eye(10)[labels_to_submit]
    one_hot_labels = one_hot_labels.reshape(-1).tolist()
    with open(p0_path, 'w') as f:
        f.write(' '.join(map(lambda x : f"{x:.6f}", points_1d)))
        f.write(' ')
        f.write(' '.join(map(lambda x : f"{x:.6f}", one_hot_labels)))
        f.write('\n')
    outputs = np.array(outputs).reshape(-1).tolist()
    with open(p1_path, 'w') as f:
        f.write(' '.join(map(lambda x : f"{x:.6f}", outputs)))
        f.write('\n')
    # Run MPC
    print("Running MPC...")
    os.system("cd ../../MP-SPDZ/ && Scripts/spdz2k.sh multi_point_val")    
        
    

import asyncio
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with specified model name")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    args = parser.parse_args()
    asyncio.run(run_experiment(args.model_name))