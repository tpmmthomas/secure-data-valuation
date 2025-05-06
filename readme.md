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

There are three notebook examples that correspond to our protocols $$\Pi_\mathsf{SS}$$, $$\Pi_\mathsf{SM}$$ and $$\Pi_\mathsf{MM}$$. You can run the notebooks in the `notebooks` directory.  

## Experiments

### Feasibility tests

Feasibility tests for our three protocols can be run inside the respective folders for the protocol in the `experiments` directory. The results will be saved in the `results` folder. 

To start the experiments, first prepare a model in `data/model.pth`, data in `data/data.pth` and label in `data/label.pth`. The formats of these files varies with your model - an example can be found in the jupyter notebooks.

Then, you can set relevant parameters in the top of the `experiment_feasible.py` file. The parameters include the models to test, the random seed, representative set size etc. 

Finally, run the following command to start the experiments:

```bash
python experiment_feasible.py
```

Note that for semi-honest single protocol, you should run with the `crypten` environment. For malicious  protocol, you should run with the `ezkl` environment.  

### Scalability tests

Scalability test is found under the `malicious_multi` folder within experiments. To run this, set the range of parameters you want to test. Then call:

```bash
python experiment_scalable.py
```

### Precision tests

This test is found under the `malicious_multi` folder within experiments. It checks if the precision of the output affects the ranking of scores between different datasets. To run this:

```bash
python experiment_precisionEffect.py
```

### Comparison with Active learning methods

This experiment is in `experiments/experiment_AL.py`. We have defined multiple valuation algorithms in the file `valuation_alg.py`, and you can add your own based on the format specified in the file.

To run the experiment, set the parameters at the top of the file and run:

```bash
python experiment_AL.py
```
