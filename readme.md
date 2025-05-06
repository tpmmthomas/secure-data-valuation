# Secure Data Valuation

This repository contains code implementation of our paper. We propose different solutions for data valuation under four different assumptions:

1. Semi-honest parties, single data-point valuation
2. Malicious parties, single data-point valuation
3. Malicious parties, multiple data-point valuation


## Environment set up

We have different environments for notebook examples and experiments.

### Notebook examples

For the notebook examples, we have provided a `environment_nb.yml` file that contains all the necessary packages. You can create a conda environment and install the requirements using the following commands:

```bash
conda env create --file environment_nb.yml
conda activate sdv2
```

### Experiments

There are 2 environments for the experiments. One is for the semi-honest, single-point scenario which uses the CrypTen library, and the other is for the malicious scenarios which uses the EZKL library.

- The crypten environment:

```bash
conda create -n crypten python=3.9 -y 
conda activate crypten
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -r requirements_crypten.txt
```

- The ezkl environment:

```bash
conda create -n ezkl python=3.9 -y
conda activate ezkl
pip install -r requirements_ezkl.txt
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

There are three notebook examples that correspond to our protocols $$\Pi_\mathsf{SS}$$, $$\Pi_\mathsf{SM}$ and $$\Pi_\mathsf{MM}$$. You can run the notebooks in the `notebooks` directory. 
