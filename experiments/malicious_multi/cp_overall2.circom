pragma circom 2.0.0;

// We need comparators for “<” and “==”
include "node_modules/circomlib/circuits/comparators.circom";
include "node_modules/circomlib/circuits/pedersen.circom";

// -------------------------------------------------------------------
// Simple Selector: picks one element out of an array based on secret idx
// -------------------------------------------------------------------
template Selector(n) {
    signal input in[n];      // public array
    signal input sel;        // private index
    signal output out;       // chosen element

    component isEq[n];
    signal indicator[n];
    var i;
    for (i = 0; i < n; i++) {
        isEq[i] = IsEqual();
        isEq[i].in[0] <== sel;
        isEq[i].in[1] <== i;
        indicator[i] <== isEq[i].out;  // 1 if sel==i, else 0
    }
    // ensure exactly one indicator is 1
    signal sum[n+1];
    sum[0] <== 0;
    for (i = 0; i < n; i++) sum[i+1] <== sum[i] + indicator[i];
    sum[n] === 1;

    // out = Σ in[i]*indicator[i]
    signal temp[n+1];
    temp[0] <== 0;
    for (i = 0; i < n; i++) temp[i+1] <== temp[i] + in[i] * indicator[i];
    out <== temp[n];
}

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

// -------------------------------------------------------------------
// Main circuit: n points, dimension = dim
// -------------------------------------------------------------------
template DistanceProof(n, dim, bitlen) {
    // public inputs
    signal input points[n][dim];   // the list of n public points
    signal input d;               // public threshold (fixed‑point)
    signal input commitX;
    signal input commitY;

    // private witness
    signal input idx;     // which point we “select”
    signal input arrBits[dim][bitlen]; //the true point in D_B
    signal input randBits[bitlen]; // the randomness in commiting D_B
    
    //Retrieve the x value
    signal x[dim];
    signal arr[dim][bitlen+1];
    for (var i = 0; i < dim; i++) {
        arr[i][0] <== 0;
        // arr[i] = Σ arrBits[i][j] * 2^j
        for (var j = 0; j < bitlen; j++) {
            arr[i][j+1] <== arr[i][j] + arrBits[i][j] * (1 << j);
        }
        x[i] <== arr[i][bitlen];
    }

    // ---- select y = points[idx] ----
    signal y[dim];
    component sel[dim];
    var i; var j;
    for (i = 0; i < dim; i++) {
        sel[i] = Selector(n);
    }
    for (i = 0; i < dim; i++) {
        // wire up the i‑th coordinate of each point
        for (j = 0; j < n; j++) {
            sel[i].in[j] <== points[j][i];
        }
        sel[i].sel <== idx;
        y[i] <== sel[i].out;
    }

    // ---- compute squared distance  Σ (x[i]–y[i])²  ----
    signal diff[dim];
    signal dist2[dim+1];
    dist2[0] <== 0;
    for (i = 0; i < dim; i++) {
        diff[i] <== x[i] - y[i];
        dist2[i+1] <== dist2[i] + diff[i] * diff[i];
    }

    // ---- enforce dist2 < d² ----
    signal d2;
    d2 <== d * d;
    // bit‑width must cover your max possible value
    component cmp = LessThan(64);
    cmp.in[0] <== dist2[dim];
    cmp.in[1] <== d2;
    cmp.out === 1;
}

// --- instantiate for your N points (replace 4 with your actual N) ---
component main {public [points,d, commitX, commitY]} = DistanceProof(20, 50, 32);
