# poseidon_iterative_modp.mpc

# Use MP‑SPDZ’s secret integer type (sint) to represent finite‐field elements.
from Compiler.types import sint, Array, Matrix
from Compiler.assembler import print_ln
from Compiler.library import for_range
import math

# -----------------------------------------------------------------------------
# Parameters (adjust these for your desired Poseidon instantiation)
# -----------------------------------------------------------------------------
# t: Poseidon state width (number of field elements in state)
t = 3  
# L: block size used in iterative hashing (each block is a vector of L field elements)
L = 2  
# For the fixed‐length hash, we use a three‐phase round structure:
# (first half of full rounds) + (partial rounds) + (second half of full rounds) + (extra round)
full_rounds_total = 4   # total full rounds; we split as half_full = full_rounds_total // 2
partial_rounds = 8      # number of partial rounds
half_full_round = full_rounds_total // 2

# S‑box exponent (for Poseidon the S‑box is x^alpha)
sbox_exponent = 5

# For demonstration we use placeholder round constants and an MDS matrix.
# (In production, use securely generated parameters.)
# round_constants is a 2D array with one row per round.
# Total rounds = full_rounds_total + partial_rounds + full_rounds_total + 1 extra round.
total_rounds = full_rounds_total + partial_rounds + full_rounds_total + 1
# We represent round constants as an Array of Arrays of sint (size: total_rounds x t).
round_constants = Array(total_rounds, Array(t, sint))
for r in range(total_rounds):
    for j in range(t):
        # For demonstration, we set all round constants to 0.
        # (In a secure instantiation these must be nonzero and fixed.)
        round_constants[r][j] = sint(0)

# MDS matrix: here we use the identity matrix as a placeholder.
MDS = Matrix(t, t, sint)
for i in range(t):
    for j in range(t):
        if i == j:
            MDS[i][j] = sint(1)
        else:
            MDS[i][j] = sint(0)

# -----------------------------------------------------------------------------
# Helper functions for Poseidon permutation (using finite field arithmetic)
# -----------------------------------------------------------------------------
def mix_with_mds(state):
    """Multiply the state vector by the MDS matrix."""
    new_state = Array(t, sint)
    for i in range(t):
        acc = sint(0)
        for j in range(t):
            acc = acc + MDS[i][j] * state[j]
        new_state[i] = acc
    return new_state

def poseidon_full_rounds(state, num_rounds, rc_counter):
    """Apply num_rounds full rounds.
    In a full round: for every state element, add a round constant and apply the S-box, then mix via MDS."""
    for r in range(num_rounds):
        for i in range(t):
            state[i] = state[i] + round_constants[rc_counter][i]
        rc_counter += 1
        for i in range(t):
            # Compute state[i]^sbox_exponent (using the built‑in exponentiation)
            state[i] = state[i] ** sbox_exponent
        state = mix_with_mds(state)
    return state, rc_counter

def poseidon_partial_rounds(state, num_rounds, rc_counter):
    """Apply num_rounds partial rounds.
    In each partial round: add round constants to all state elements, then apply S-box only to the first element, then mix."""
    for r in range(num_rounds):
        for i in range(t):
            state[i] = state[i] + round_constants[rc_counter][i]
        rc_counter += 1
        # Only the first element goes through the S-box.
        state[0] = state[0] ** sbox_exponent
        state = mix_with_mds(state)
    return state, rc_counter

def poseidon_hash_fixed(block):
    """
    Computes the Poseidon hash for a fixed-length input block.
    'block' is a list (or Array) of L sint elements.
    The function pads the block to length t (with zeros) if needed, then applies:
      full rounds (first half), partial rounds, full rounds (second half),
      one extra S-box round, and one final MDS multiplication.
    Finally, the output is taken as state[1].
    """
    # Initialize state: pad block with zeros until length t.
    state = Array(t, sint)
    for i in range(t):
        if i < len(block):
            state[i] = block[i]
        else:
            state[i] = sint(0)
    
    rc_counter = 0  # round constant index

    # First full rounds (half)
    state, rc_counter = poseidon_full_rounds(state, half_full_round, rc_counter)
    # Partial rounds
    state, rc_counter = poseidon_partial_rounds(state, partial_rounds, rc_counter)
    # Second full rounds (half)
    state, rc_counter = poseidon_full_rounds(state, half_full_round, rc_counter)
    # Extra round: apply S-box to all elements and then mix with MDS.
    for i in range(t):
        state[i] = state[i] ** sbox_exponent
    state = mix_with_mds(state)
    # Return the hash as the second element of the state (state[1])
    return state[1]

# -----------------------------------------------------------------------------
# Iterative Poseidon Hashing (Merkle Tree style)
# -----------------------------------------------------------------------------
def iterative_poseidon_hash(message):
    """
    Computes the Poseidon hash of an arbitrarily long message using iterative (Merkle tree–style) hashing.
    'message' is a list of sint elements.
    
    The algorithm:
      1. Set hash_inputs = message.
      2. While len(hash_inputs) > 1 or not one_iter:
           - For each consecutive chunk (of length L) of hash_inputs (pad the last chunk with zeros if needed),
             compute block_hash = poseidon_hash_fixed(chunk).
           - Set hash_inputs = list of these block_hash values.
           - Set one_iter to True.
      3. Return hash_inputs.
      
    This mimics the Rust code:
        while hash_inputs.len() > 1 || !one_iter { ... } 
    """
    # Start with the original message (a Python list of sint)
    hash_inputs = message[:]  # copy
    one_iter = False

    # Continue until a single hash remains (or at least one iteration is done)
    while (len(hash_inputs) > 1) or (not one_iter):
        new_hashes = []  # will hold the hash of each block
        num_blocks = (len(hash_inputs) + L - 1) // L  # ceiling division
        # Process each block of length L
        for block_idx in range(num_blocks):
            # Determine the slice indices
            start_idx = block_idx * L
            end_idx = start_idx + L
            # Copy the block; if the block is shorter than L, pad with zeros.
            block = []
            for i in range(L):
                msg_idx = start_idx + i
                if msg_idx < len(hash_inputs):
                    block.append(hash_inputs[msg_idx])
                else:
                    block.append(sint(0))
            # Compute the Poseidon hash for this fixed-length block.
            block_hash = poseidon_hash_fixed(block)
            new_hashes.append(block_hash)
        one_iter = True
        hash_inputs = new_hashes  # update for next iteration

    # Return the final hash vector (in Rust they return vec![hash_inputs])
    return hash_inputs

# -----------------------------------------------------------------------------
# Main function: load inputs and verify the hash
# -----------------------------------------------------------------------------
def poseidon_verify_iterative():
    """
    Reads a longer message from one party (Party 0) and a claimed hash from another (Party 1),
    computes the iterative Poseidon hash and compares it with the claimed hash.
    """
    MESSAGE_PROVIDER = 0  # Party 0 provides the message
    HASH_PROVIDER = 1     # Party 1 provides the claimed hash

    # For this example, we fix a maximum message length.
    MAX_MESSAGE_LENGTH = 6  # (Adjust as needed)
    
    # Load the message elements as sint values from Party 0.
    message = []
    for i in range(MAX_MESSAGE_LENGTH):
        # Here we assume all MAX_MESSAGE_LENGTH inputs are provided.
        message.append(sint.get_input_from(MESSAGE_PROVIDER, i))
    
    # Load the claimed hash from Party 1.
    claimed_hash = sint.get_input_from(HASH_PROVIDER, 0)
    
    # Compute the iterative Poseidon hash of the message.
    computed_hashes = iterative_poseidon_hash(message)
    # (By design the loop stops when only one hash remains.)
    computed_hash = computed_hashes[0]
    
    # Compare computed hash with the claimed hash.
    # (In MP‑SPDZ, equality returns a secret bit; we then reveal it.)
    equal = (computed_hash - claimed_hash) == sint(0)
    result = equal.reveal()
    print_ln("Hash Verification Result: %s", result)
    # For debugging, you might also reveal the computed hash:
    print_ln("Computed Hash: %s", computed_hash.reveal())

# -----------------------------------------------------------------------------
# Execute the verification function.
# -----------------------------------------------------------------------------
poseidon_verify_iterative()
