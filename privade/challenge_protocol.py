import os
import json

cur_dir = os.path.dirname(os.path.abspath(__file__))

def setup_challenge_protocol():
    # Create necessary directories
    os.system(f"cd {cur_dir}/zk_helpers && circom cp_overall.circom --r1cs --wasm --sym")
    os.system(f"cd {cur_dir}/zk_helpers && snarkjs powersoftau new bn128 18 pot18_0000.ptau -v")
    os.system(f'cd {cur_dir}/zk_helpers && echo "random_string" | snarkjs powersoftau contribute pot18_0000.ptau pot18_0001.ptau --name="First contribution" -v')
    os.system(f"cd {cur_dir}/zk_helpers && snarkjs powersoftau prepare phase2 pot18_0001.ptau pot18_final.ptau -v")
    os.system(f"cd {cur_dir}/zk_helpers && snarkjs groth16 setup cp_overall.r1cs pot18_final.ptau cp_0000.zkey")
    os.system(f'cd {cur_dir}/zk_helpers && echo "random" | snarkjs zkey contribute cp_0000.zkey cp_0001.zkey --name="1st Contributor Name" -v')
    os.system(f"cd {cur_dir}/zk_helpers && snarkjs zkey export verificationkey cp_0001.zkey verification_key.json")

def create_proof(cp_data,proof_file):
    with open(f'{cur_dir}/zk_helpers/cp.json', 'w') as f:
        json.dump(cp_data, f)
    exit_code = os.system(f"cd {cur_dir}/zk_helpers && node commit.js")
    assert exit_code == 0, "Command 'node commit.js' failed"
    # Generate the witness
    exit_code = os.system(f"cd {cur_dir}/zk_helpers && node cp_overall_js/generate_witness.js cp_overall_js/cp_overall.wasm input.json witness.wtns")
    assert exit_code == 0, "Command to generate witness failed"
    # Generate the proof
    exit_code = os.system(f"cd {cur_dir}/zk_helpers && snarkjs groth16 prove cp_0001.zkey witness.wtns {proof_file} public.json")
    assert exit_code == 0, "Command to generate proof failed"
    return True
    
def verify_proof(proof_file):
    exit_code = os.system(f"cd {cur_dir}/zk_helpers && snarkjs groth16 verify verification_key.json public.json {proof_file}")
    assert exit_code == 0, "Command to verify proof failed"
    return True
    
    
    

if __name__ == "__main__":
    print(cur_dir)