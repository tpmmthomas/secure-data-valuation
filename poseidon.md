Below is a step-by-step outline for recreating the Poseidon hash offline, using only the files that `ezkl` produces during your end-to-end proof flow. 

---

## Overview

1. **Identify (and extract) the private preimage from the witness**  
2. **Identify (and extract) the hashed result from the circuit’s public outputs**  
3. **Obtain the exact Poseidon parameters (round constants, MDS matrix, rate, capacity, etc.)** used by `ezkl`  
4. **Run a Poseidon library offline** with the same parameters on the extracted preimage, and compare the result with the hashed output that appears as a public value.

Because your `output_visibility` is `hashed/public`, `ezkl` places the Poseidon digest in the *public outputs* (public instance data), while the un-hashed “raw” output is in the private witness. Below, we break down how to find both pieces.

---

## 1. Extracting the Private Preimage from the Witness

### The witness file

After running:

```bash
ezkl gen_witness \
  --data <data_path> \
  --model <compiled_model_path> \
  --output <witness_path> \
  --settings <settings_path>
```

you get a JSON or binary file at `<witness_path>`. That file contains all the *private* circuit assignments (i.e. the “advice” columns in Halo2-speak). For a neural-network model, it typically includes:

- Input activations (especially if `input_visibility` was `private`, or partial if `public`)  
- Intermediate activations from each layer  
- Final layer’s raw output **before** hashing  

**However, the witness file is not always super-friendly** to read. It’s a large array of field elements, often with minimal labeling. There are a few ways you can proceed:

1. **`ezkl parse-witness` (if available)**  
   - *As of `ezkl` 0.8.x*, there is an (experimental) command to parse or dump the witness in a more structured way.  
   - Run `ezkl parse-witness --help` to see if your version supports this.  
   - It may produce a table or JSON with all the internal columns.  
   
2. **Manually interpret the JSON**  
   - If your witness is JSON-formatted, you can open it up and see a top-level structure (like `{"witness": [[...col0...],[...col1...], ...]}`).  
   - The last columns/rows often correspond to the final outputs, which are then fed into a Poseidon gadget.  

3. **`ezkl table` or `ezkl mock`** in debug mode  
   - Running `ezkl mock ...` sometimes gives a debug “trace” of the circuit with row-by-row assignments.  
   - This can help you see which row or column is the final output preimage.  

### Locating the final output

- If the network is a typical single-output model, you’ll often find *one or a small handful* of final layer outputs.  
- These final layer outputs (one or more field elements) then become the *input* to the Poseidon chip.  
- Typically, the Poseidon gadget is configured to produce a single field element digest, which is the hashed public output.

Once you identify *which part* of the witness is the final (unhashed) output, you can copy that numeric value. (If your model has multiple outputs, you might see multiple “preimages” that get hashed into a single digest or multiple digests, depending on how `ezkl` is configured.)

---

## 2. Extracting the Hashed Result from the Public Instance

Because you set `output_visibility = "hashed/public"`, the final Poseidon hash (digest) is placed in the circuit’s *public instance*. That means:

- It will appear in the JSON (or other data structure) of the *public inputs / instance* array that `ezkl` generates for verification.  
- If you are verifying on-chain, that same digest would typically appear on-chain as well.

A couple ways to see it:

1. **Look at the proof JSON** (if your proof is in a JSON-like format). Sometimes, `ezkl` prints a snippet like:

   ```json
   "instances": [
     ["<digest_field_element>", ...]
   ]
   ```
   
2. **Look at the standard out** of `ezkl prove` or `ezkl verify`—some versions print the public inputs.  

One way or another, you should see a field element that looks like `0x097bb7b76f35fc78c390f0...`. That is the Poseidon hash in the finite field for BN254 (or whichever curve you are using).

---

## 3. Getting the Poseidon Parameters

### Where do these come from?

`ezkl` uses a “Poseidon Chip” under the hood. The parameters (round constants, MDS matrix, etc.) are *hard-coded* for each configuration: the rate, capacity, number of rounds, etc. If you used the default (and typical) BN254 Poseidon with width=3 or width=4, `ezkl` is pulling in default parameters from within its Rust library.  

**They are not stored in the witness** because they’re part of the circuit design, not user data. That means you need to replicate the same parameters offline to get the same hash result.

### Obtaining them

There are a few ways to get the exact constants:

1. **Look at `ezkl`’s source code** for the version you’re using. For example, in GitHub you’ll find something like `poseidon/mod.rs` or `circuit/poseidon.rs` where the parameters are defined for BN254. If you see lines like:

   ```rust
   pub const ROUND_CONSTANTS: [Felt; <some_number>] = [ ... ];
   pub const MDS_MATRIX: [[Felt; RATE]; RATE] = ...
   ```
   then that’s your MDS matrix and round constants.  

2. **Use an external Poseidon library** that matches the standard “Ethereum-compatible” BN254 Poseidon spec.  
   - Many libraries (e.g. `poseidon-rs`, `circomlib` Poseidon, or a `poseidon_py` library) have the standard sets of parameters.  
   - You must ensure *exactly* the same Poseidon variant: same `alpha` (the exponent in S-Box), same number of `full_rounds`, `partial_rounds`, etc., same rate/capacity.  

When in doubt, the simplest path is usually to read the `ezkl` source for your release tag to confirm the parameters.  

---

## 4. Recompute the Poseidon Hash Offline

Finally, you can replicate the hash by:

1. **Convert the raw output (preimage) from decimal/hex string → field element** under BN254.  
2. **Pass that into a Poseidon “hash one field element (or N elements) → one digest”** function, with the identical constants.  
3. **Compare** the computed result with `0x097bb7b7...` (your public hashed output).

A typical Python snippet might look like:

```python
from poseidon_py import poseidon_params, poseidon_hash
# Hypothetical library name for example

# 1. Convert your preimage to a field element
# Suppose your final output from the witness was a hex string or decimal
preimage_hex = "0x1234abc..."    # or decimal
preimage_int = int(preimage_hex, 16)  # convert hex => int

# 2. Load matching Poseidon parameters
params = poseidon_params(curve="BN254",  # or the precise param set
                         width=3,        # or 2,4, etc. must match your circuit
                         full_rounds=63, # example
                         partial_rounds=22, # example
                         security=128)

# 3. Compute the Poseidon hash offline
digest = poseidon_hash([preimage_int], params)

# 4. Compare with the public instance output
print(hex(digest))
# should match "0x097bb7b76f35fc78c390f0379c176a62a7082d8315264f92f4f419b35a89dd1c"
```

*(Exact parameter names and method calls will differ based on the library you use.)*

---

## Summary

1. **Witness file →** find the *raw final output(s)* that feed Poseidon.  
2. **Public instance →** find the hashed result (Poseidon digest).  
3. **Source code (or known spec) →** gather the same Poseidon round constants, MDS matrix, rate, capacity, etc.  
4. **Run a Poseidon hash offline** on the extracted preimage.  
5. **Verify** that your offline result matches the circuit’s public hashed output.

That’s it! Once you match those values, you’ve successfully replicated the Poseidon hash exactly as computed inside `ezkl`’s zero-knowledge circuit.