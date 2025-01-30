# Secure Data Valuation

This repository contains code implementation of our paper. We propose different solutions for data valuation under four different assumptions:

1. Semi-honest parties, single data-point valuation
2. Semi-honest parties, multiple data-point valuation
3. Malicious parties, single data-point valuation
4. Malicious parties, multiple data-point valuation


## Environment set up
```bash
conda create -n secdataval python=3.11 -y 
conda activate secdataval
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -r requirements.txt
```

Additionally, for scenarios that involve MPC protocols with the MP-SPDZ library, you need to be on the linux machine and run the following installation:

```bash
sudo apt-get install automake build-essential clang cmake git libboost-dev libboost-filesystem-dev libboost-iostreams-dev libboost-thread-dev libgmp-dev libntl-dev libsodium-dev libssl-dev libtool python3
cd MP-SPDZ
make setup
make -j8 mascot-party.x
```

## Running the examples

Simply go to the folder of the scenario you want to run `[single-point/multi-point]/[semi-honest/malicious]` and launch the `example.ipynb` notebook. Detailed code and explanations are provided in the notebook.
