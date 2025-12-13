# Secure Data Valuation

This repository contains code implementation of our paper.

## Environment set up

We have provided a `environment_nb.yml` file that contains all the necessary packages. You can create a conda environment and install the requirements using the following commands:

```bash
conda env create --file environment_nb.yml
conda activate sdv2
```

### MP-SPDZ setup

Additionally, for scenarios that involve MPC protocols with the MP-SPDZ library, you need to be on the linux machine and run the following installation:

```bash
sudo apt-get install automake build-essential clang cmake git libboost-dev libboost-filesystem-dev libboost-iostreams-dev libboost-thread-dev libgmp-dev libntl-dev libsodium-dev libssl-dev libtool python3
cd MP-SPDZ
make setup
make -j8 mascot-party.x
make -j8 spdz2k-party.x
```

### Circom and snarkjs setup

For the circom setup, you need to install rust, the circom and snarkjs library. You can do this by running the following command:

```bash
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
git clone https://github.com/iden3/circom.git
cd circom
cargo build --release
cargo install --path circom
npm install -g snarkjs 
```

Moreover, you will need to install the relevant javascript packages. In directories where `package.json` is present, run the following command:

```bash
npm install
```


## Running the notebook examples

The notebook example in the `notebooks` directory takes you through the whole process of PrivaDE.

## Experiments

### Feasibility tests

The feasibility test which evaluates our methods can be found in `experiments/benchmark_privade.py`.

```bash
python experiment_feasible.py
```

Note that for semi-honest single protocol, you should run with the `crypten` environment. For malicious  protocol, you should run with the `ezkl` environment.  

### Comparison with Active learning methods

This experiment is in `experiments/experiment_AL.py`. We have defined multiple valuation algorithms in the file `valuation_alg.py`, and you can add your own based on the format specified in the file.

To run the experiment, set the parameters at the top of the file and run:

```bash
python experiment_AL.py
```

### Model Definitions
For model definitions used in our paper, see [here](model_def.md)
