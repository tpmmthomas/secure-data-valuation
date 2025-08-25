import os
import numpy as np

#Current directory
cur_dir = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(f'{cur_dir}/../MP-SPDZ/Player-Data'):
    os.makedirs(f'{cur_dir}/../MP-SPDZ/Player-Data')
p0_path = os.path.join(f'{cur_dir}/../MP-SPDZ/Player-Data','Input-P0-0')
p1_path = os.path.join(f'{cur_dir}/../MP-SPDZ/Player-Data','Input-P1-0')

def prepare_inputs(bob_input, alice_input,size=10):
    #Unpack Bob's input
    points_to_submit, labels_to_submit = bob_input

    #Turn points to submit into a 1D list
    points_1d = np.array(points_to_submit).reshape(-1).tolist()
    print(len(points_1d))
    #Convert labels into one-hot encoding
    one_hot_labels = np.eye(size)[labels_to_submit]
    print(one_hot_labels.shape)
    one_hot_labels = one_hot_labels.reshape(-1).tolist()
    print(len(one_hot_labels))
    with open(p0_path, 'w') as f:
        f.write(' '.join(map(lambda x : f"{x:.6f}", points_1d)))
        f.write(' ')
        f.write(' '.join(map(lambda x : f"{x:.6f}", one_hot_labels)))
        f.write('\n')
    #Alice's input
    outputs = np.array(alice_input).reshape(-1).tolist()
    print(len(outputs))
    with open(p1_path, 'w') as f:
        f.write(' '.join(map(lambda x : f"{x:.6f}", outputs)))
        f.write('\n')
    return True

def compile_program():
    exit_code = os.system(f"cd {cur_dir}/../MP-SPDZ && ./compile.py multi_point_val -R 64")
    return exit_code == 0

def run_mpc():
    exit_code = os.system(f"cd {cur_dir}/../MP-SPDZ/ && Scripts/spdz2k.sh multi_point_val -v")
    return exit_code == 0