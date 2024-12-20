# Single point protocol

TODO: add commitments.

## Semi-honest scenario

In the semi-honest scenario, x,y,M are all protected.

1. Alice and Bob runs MPC inference using Crypten.
2. Bob receives y', then computes loss and sends to Alice.

## Malicious scenario

In the malicious scenario, only y, M are protected.

1. Bob sends x to Alice.
2. Alice run model inference with ZKP
3. Alice sends y' to Bob.
4. Bob computes loss with ZKP and sends to Alice.

Variation: If there exists a MPC-in-the-cloud with honest majority, can run cloud inference. Secret share inputs.

### TODO

- [ ] Add Commitments  
