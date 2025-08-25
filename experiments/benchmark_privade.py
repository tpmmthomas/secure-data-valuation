import os
import torch
import sys
import random
sys.path.append('..')  # Add privade directory to path
from privade.data import get_dataset
from privade.models import get_model
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 1000 #Bob's dataset size

full_model = get_model('vgg8', 'cifar100')
full_data = get_dataset('cifar100')

# Randomly select 1000 images as Bob's dataset
indices = random.sample(range(len(full_data)), N)
bob_images = np.array([full_data[i][0].numpy() for i in indices])
bob_labels = np.array([full_data[i][1]  for i in indices])

# Ramdomly select 1000 images as Alice's initial dataset
indices = random.sample(range(len(full_data)), N)
alice_images = np.array([full_data[i][0].numpy() for i in indices])
alice_labels = np.array([full_data[i][1]  for i in indices])

#Randomly select 100 images as test set
indices = random.sample(range(len(full_data)), 100)
test_images = np.array([full_data[i][0].numpy() for i in indices])
test_labels = np.array([full_data[i][1]  for i in indices])

# Train the model for a few epochs
alice_images_tensor = torch.FloatTensor(alice_images)
alice_labels_tensor = torch.LongTensor(alice_labels)
alice_dataset = TensorDataset(alice_images_tensor, alice_labels_tensor)
alice_dataloader = DataLoader(alice_dataset, batch_size=32, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(full_model.parameters(), lr=0.001)
full_model.train()
for epoch in range(10):
    running_loss = 0.0
    for images, labels in alice_dataloader:
        optimizer.zero_grad()
        outputs = full_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/5, Loss: {running_loss/len(alice_dataloader):.4f}')
    
#Test model
full_model.eval()
test_images_tensor = torch.FloatTensor(test_images)
test_labels_tensor = torch.LongTensor(test_labels)
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        outputs = full_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# Save models and datasets
torch.save(bob_images, 'data/bob_images.pth')
torch.save(bob_labels, 'data/bob_labels.pth')
torch.save(full_model.state_dict(), 'data/alice_full_model.pth')

from privade.distillation import train_distilled_model

student_model = get_model('cifarcnn5', 'cifar100')

kd_train_loader = DataLoader(alice_dataset, batch_size=32, shuffle=True)

trained_student_model = train_distilled_model(full_model, student_model, kd_train_loader, test_dataloader,epochs=10)

torch.save(trained_student_model.state_dict(), "data/alice_student_model.pth")


from privade.split import split_model
try:
    
    # Split the model
    model_A, model_B, model_C, split_stats = split_model(
        data_loader=alice_dataloader,
        model=trained_student_model,
    )
    model_A.to(device)
    model_B.to(device)
    model_C.to(device)

    print(f"\nSplit successful!")
    print(f"First activation layer: {split_stats['first_activation_layer']}")
    print(f"Optimal boundary layer: {split_stats['optimal_layer']}")
    print(f"Privacy preserved rate: {split_stats['privacy_preserved_rate']:.3f}")
    print(f"Client model (model_B): {len(list(model_B.children()))} layers")
    print(f"Server model (model_C): {len(list(model_C.children()))} layers")
    
except Exception as e:
    print(f"Error during splitting: {e}")
    import traceback
    traceback.print_exc()
    
torch.save(model_A.state_dict(),"data/model_a.pth")
torch.save(model_B.state_dict(),"data/model_b.pth")
torch.save(model_C.state_dict(),"data/model_c.pth") 

from torchsummary import summary
print(summary(model_A, input_size=alice_images[0].shape))

from privade.dim_reduction import reduce_image_dimensions

target_dimension = 50
reduced_images, _,_ = reduce_image_dimensions(bob_images, target_dimension)

torch.save(reduced_images, 'data/reduced_images.pth')
torch.save(bob_labels, 'data/reduced_labels.pth')

#Perform clustering
from privade.clustering import kmeans_clustering

rep_set_size = 50

start = time.time()
representative_set = kmeans_clustering(reduced_images, rep_set_size)
end = time.time()
representative_points = reduced_images[representative_set]
rep_points = bob_images[representative_set]
rep_labels = bob_labels[representative_set]
torch.save(rep_points, 'data/rep_points.pth')
torch.save(rep_labels, 'data/rep_labels.pth')
dists = np.linalg.norm(reduced_images[:, None] - representative_points[None, :], axis=2)
min_dists = np.min(dists, axis=1)
max_min_distance = np.ceil(np.max(min_dists))
print("Maximum of the minimum distances:", max_min_distance)

from privade.challenge_protocol import setup_challenge_protocol
from benchmark_util import measure_peak_rss, measure_peak_rss_safe

# CP_memory = 0 
# start = time.time()
# info = measure_peak_rss(setup_challenge_protocol)
# end = time.time()
# print("Challenge Protocol setup time: ", end - start)
# print(f"Peak RSS: {info['peak_rss_bytes']/1e6:.1f} MB")
# CP_memory += info['peak_rss_bytes'] #Peak RSS: 1150.9 MB

# def cp_main():

#     from privade.challenge_protocol import create_proof, verify_proof

#     M = 20 #challenge number

#     #Alice randomly selects M points from the whole dataset
#     indices = random.sample(range(N), M)

#     #For each data point do the Challenge Protocol
#     total_com_size = 0

#     for idx in indices:
#         #Find the index from representative_points which has the min distance from the selected point
#         selected_point = reduced_images[idx]
#         dists = np.linalg.norm(representative_points - selected_point, axis=1)
#         min_index = np.argmin(dists)
#         print(np.min(dists), min_index)
#         cp_data = {
#             "messageArray": selected_point.tolist(),
#             "idx": int(min_index),
#             "allPoints": representative_points.tolist(),
#             "d": int(max_min_distance),
#             "r": 0x12345678
#         }
#         assert len(selected_point.tolist()) == 50
#         assert len(representative_points.tolist()) == 50
        
#         proof_file = "proof.json"
        
#         assert create_proof(cp_data,proof_file)
        
#         assert verify_proof(proof_file)
        
#         total_com_size += os.path.getsize("/home/thomas/secure-data-valuation/privade/zk_helpers/proof.json")

#     print("Total communication size:", total_com_size)

# start = time.time()
# info = measure_peak_rss(cp_main)
# end = time.time()
# print("Challenge Protocol run time: ", end - start)
# print(f"Peak RSS: {info['peak_rss_bytes']/1e6:.1f} MB")
# CP_memory += info['peak_rss_bytes']


model_A = model_A.to(device)
model_A_output = model_A(torch.tensor(rep_points).to(device))

from privade.cnczk import choose_random_layers, collect_sequential_activations, get_layer, setup_zkp
import asyncio
#Plaintext inference
model_B.to(device)
model_B_output = model_B(model_A_output)

def get_all_modules(module):
    """Recursively get all modules in order."""
    modules = []
    for child in module.children():
        if isinstance(child, nn.Sequential):
            # Flatten Sequential containers
            modules.extend(get_all_modules(child))
        elif len(list(child.children())) == 0:
            # Leaf module
            modules.append(child)
        else:
            # Intermediate module with children
            modules.extend(get_all_modules(child))
    return modules

total_layers = len(get_all_modules(model_B))
num_layers = int(total_layers // 2)

print("B has layers:", total_layers)
print("Total layers verified for B: ", num_layers)
#Randomly choose layers:
layers = choose_random_layers(model_B,num_layers)

def setup_zkp_b():
    # Ensure models and data are completely on CPU and detached from CUDA
    cpu_model_B = model_B.cpu()
    cpu_model_A_output = model_A_output.cpu().detach().clone()
    
    # Clear any CUDA cache before subprocess
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for layer in layers:
        asyncio.run(setup_zkp(cpu_model_B, cpu_model_A_output, layer, 'pw'))

import tracemalloc
from benchmark_util import PeakRSS
    
# with  PeakRSS(interval=0.002) as mw:
#     start = time.time()
#     # Use CUDA-safe memory measurement
#     setup_zkp_b()
#     end = time.time()
    
# print("Total time: ",end-start)
# print("PEak memory: ", mw.peak_rss/1024/1024)


    
import random
# Alice randomly chooses s number of points
s = 30
points_to_check = random.sample(range(len(model_A_output)), s)
selected_A_output = model_A_output[points_to_check]
selected_A_output.shape

from privade.cnczk import prove_zkp, verify_zkp

print("Proving B:")

def proof_zkp_b():
    total_proof_size = 0
    cpu_model_B = model_B.cpu()
    cpu_model_A_output = model_A_output.cpu().detach().clone()
    for layer in layers:
        asyncio.run(prove_zkp(cpu_model_B,cpu_model_A_output,layer))

    for layer in layers:
        total_proof_size += os.path.getsize(f"/home/thomas/secure-data-valuation/experiments/data/layer_{layer}_test.pf")
        total_proof_size += os.path.getsize(f"/home/thomas/secure-data-valuation/experiments/data/layer_{layer}_test.vk")
        asyncio.run(verify_zkp(layer))
    print("Total proof size: ", total_proof_size)

# with  PeakRSS(interval=0.002) as mw:
#     start = time.time()
#     proof_zkp_b()
#     end = time.time()
# print("B proof run time: ", end - start)
# print(f"Peak RSS: {mw.peak_rss/1024/1024} MB")
        
model_C.to(device)
model_C_output = model_C(model_B_output)

total_layers = len(get_all_modules(model_C))
num_layers = int(total_layers // 2)

print("C has layers:", total_layers)
print("Total layers verified for C: ", num_layers)
#Randomly choose layers:
layers = choose_random_layers(model_C,num_layers)

def setup_zkp_c():
    # Ensure models and data are completely on CPU and detached from CUDA
    cpu_model_C = model_C.cpu()
    cpu_model_B_output = model_B_output.cpu().detach().clone()
    
    # Clear any CUDA cache before subprocess
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for layer in layers:
        print(f"Setting up layer {layer}")
        asyncio.run(setup_zkp(cpu_model_C, cpu_model_B_output, layer, 'pi'))

import tracemalloc
from benchmark_util import PeakRSS
    
# with  PeakRSS(interval=0.002) as mw:
#     start = time.time()
#     # Use CUDA-safe memory measurement
#     setup_zkp_c()
#     end = time.time()
    
# print("Total time: ",end-start)
# print("PEak memory: ", mw.peak_rss/1024/1024)


    
# import random
# # Alice randomly chooses s number of points
# s = 30
# points_to_check = random.sample(range(len(model_A_output)), s)
# selected_A_output = model_A_output[points_to_check]
# selected_A_output.shape

# from privade.cnczk import prove_zkp, verify_zkp

# print("Proving C:")

# def proof_zkp_c():
#     total_proof_size = 0
#     cpu_model_C = model_C.cpu()
#     cpu_model_B_output = model_B_output.cpu().detach().clone()
#     for layer in layers:
#         asyncio.run(prove_zkp(cpu_model_C,cpu_model_B_output,layer))

#     for layer in layers:
#         total_proof_size += os.path.getsize(f"/home/thomas/secure-data-valuation/experiments/data/layer_{layer}_test.pf")
#         total_proof_size += os.path.getsize(f"/home/thomas/secure-data-valuation/experiments/data/layer_{layer}_test.vk")
#         asyncio.run(verify_zkp(layer))
#     print("Total proof size: ", total_proof_size)

# with  PeakRSS(interval=0.002) as mw:
#     start = time.time()
#     proof_zkp_c()
#     end = time.time()
# print("C proof run time: ", end - start)
# print(f"Peak RSS: {mw.peak_rss/1024/1024} MB")


points_to_submit = reduced_images[representative_set]
labels_to_submit = rep_labels

Bob_input = (points_to_submit, labels_to_submit)
Alice_input = model_C_output.cpu().detach()

from privade.scoring import prepare_inputs

assert prepare_inputs(Bob_input, Alice_input,size=100)

from privade.scoring import compile_program, run_mpc
assert compile_program()
assert run_mpc()