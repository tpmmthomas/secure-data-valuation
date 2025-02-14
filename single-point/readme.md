# Single point protocol

## Semi-honest scenario

In the semi-honest scenario, x,y,M are all protected.

1. Alice and Bob runs MPC inference using Crypten.
2. Bob receives y', then computes loss and sends to Alice.

To understand how these works:  
1. First go to `semi-honest/example.ipynb` to understand the protocol, and create the model architecture for Alice, the data point with label for Bob and save them to appropriate `.pth` files.
2. Then run `python semi-honest/launcher.py` to run the protocol with multi-processing to simulate the real-world execution.


### Assumed Public data
- Model architecture

### Protected data
- Model weights
- Input data
- Labels

## Malicious scenario

In the malicious scenario, only y, M are protected.

1. Bob sends x to Alice.
2. Alice run model inference with ZKP
3. Alice sends y' to Bob.
4. Bob computes loss with ZKP and sends to Alice.

(3,4 update): Alice does not send y' to Bob. Instead, Alice supply the ZKP of y' to Bob. Bob verifies the ZKP. They compute loss together with MPC protocol.

Variation: If there exists a MPC-in-the-cloud with honest majority, can run cloud inference. Secret share inputs.

### TODO

- [ ] Add Commitments  
