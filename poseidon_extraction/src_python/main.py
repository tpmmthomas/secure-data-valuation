import poseidon
from poseidon_params import MDS, RC


security_level = 128
alpha = 5
input_rate = 4
t = 2
full_round=8
partial_round=56
poseidon_new = poseidon.OptimizedPoseidon()
# poseidon_new = poseidon.Poseidon(poseidon.parameters.prime_254,security_level,alpha,input_rate,t,full_round,partial_round,MDS,RC)