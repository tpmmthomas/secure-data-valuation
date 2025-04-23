pragma circom 2.0.0;

// Import the BabyJub‐based Pedersen hash from circomlib
include "node_modules/circomlib/circuits/pedersen.circom";

template CommitmentProofArray(n, bitlen) {
    // Public inputs: Pedersen commitment point
    signal input commitX;
    signal input commitY;

    // Private inputs:
    //  - arrBits[i][j]: the jᵗʰ bit of the iᵗʰ integer
    //  - randBits[j]: the jᵗʰ bit of the blinding factor r
    signal input arrBits[n][bitlen];
    signal input randBits[bitlen];

    // 1) Enforce Boolean-ness
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < bitlen; j++) {
            arrBits[i][j] * (arrBits[i][j] - 1) === 0;
        }
    }
    for (var j = 0; j < bitlen; j++) {
        randBits[j] * (randBits[j] - 1) === 0;
    }

    // 2) Reconstruct each integer and do your extra ZK computations
    //    e.g. here we compute the sum of the array as a demo
    signal arr[n][bitlen+1];
    for (var i = 0; i < n; i++) {
        arr[i][0] <== 0;
        // arr[i] = Σ arrBits[i][j] * 2^j
        for (var j = 0; j < bitlen; j++) {
            arr[i][j+1] <== arr[i][j] + arrBits[i][j] * (1 << j);
        }
    }
    arr[1][bitlen] === 2;

    // 3) Feed all bits into one Pedersen hash
    var totalBits = n * bitlen + bitlen;
    component ped = Pedersen(totalBits);

    // copy arrBits…
    var idx = 0;
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < bitlen; j++) {
            ped.in[idx] <== arrBits[i][j];
            idx += 1;
        }
    }

    // …and then the randomness bits
    for (var j = 0; j < bitlen; j++) {
        ped.in[idx] <== randBits[j];
        idx += 1;
    }

    // 4) Enforce the public commitment equals the hash point
    commitX === ped.out[0];
    commitY === ped.out[1];
}

template CommitmentProof(msgBits, randBits) {
    // Public inputs: the two field elements making up the EC point (X,Y)
    signal input commitX;
    signal input commitY;

    // Private inputs: bit‐decompositions of the message and randomness
    signal input msg[msgBits];
    signal input rand[randBits];

    // Enforce that each is a Boolean (0 or 1)
    for (var i = 0; i < msgBits; i++) {
        msg[i] * (msg[i] - 1) === 0;
    }
    for (var j = 0; j < randBits; j++) {
        rand[j] * (rand[j] - 1) === 0;
    }

    // Build the Pedersen hash on (msg || rand)
    component ped = Pedersen(msgBits + randBits);
    for (var i = 0; i < msgBits; i++) {
        ped.in[i] <== msg[i];
    }
    for (var j = 0; j < randBits; j++) {
        ped.in[msgBits + j] <== rand[j];
    }

    // Enforce equality with the public commitment
    commitX === ped.out[0];
    commitY === ped.out[1];
}

// Instantiate with, e.g., 32‐bit message + 32‐bit randomness
component main {public [commitX, commitY]} = CommitmentProofArray(10, 32);
