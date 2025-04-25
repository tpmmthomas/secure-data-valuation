1. Compile: circom cp_overall.circom --r1cs --wasm --sym
2. Generate witness inputs: node commit.js
3. Gnerate witness: node cp_overall_js/generate_witness.js cp_overall_js/cp_overall.wasm input.json witness.wtns
4. Setup (only done once)
    - snarkjs powersoftau new bn128 16 pot12_0000.ptau -v
    - snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau --name="First contribution" -v
    - snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v
    - snarkjs groth16 setup cp_overall.r1cs pot12_final.ptau multiplier2_0000.zkey
    - snarkjs zkey contribute multiplier2_0000.zkey multiplier2_0001.zkey --name="1st Contributor Name" -v
    - snarkjs zkey export verificationkey multiplier2_0001.zkey verification_key.json

5. Generate proof: snarkjs groth16 prove multiplier2_0001.zkey witness.wtns proof.json public.json
6. Verify proof: snarkjs groth16 verify verification_key.json public.json proof.json