// commit.js
const fs = require("fs");
const circomlibjs = require("circomlibjs");

const FRACTIONAL_BITS = 16;
const SCALE = 1 << FRACTIONAL_BITS; 

// helper: decompose a 32‑bit integer into a little‑endian bit array
function toBitsLE(x, n) {
  const bits = [];
  for (let i = 0; i < n; i++) {
    bits.push((x >> i) & 1);
  }
  return bits;
}

// helper: pack an array of 0/1 bits (little‑endian) into a Buffer
function packBitsLE(bits) {
  const byteLen = Math.ceil(bits.length / 8);
  const buf = Buffer.alloc(byteLen);
  for (let i = 0; i < bits.length; i++) {
    if (bits[i] === 1) {
      const bytePos = Math.floor(i / 8);
      const bitPos  = i % 8;       // little‑endian inside each byte
      buf[bytePos] |= (1 << bitPos);
    }
  }
  return buf;
}

async function main() {
  // 1) Build the circomlibjs Pedersen hash & BabyJub context
  const pedersen = await circomlibjs.buildPedersenHash();
  const babyJub  = await circomlibjs.buildBabyjub();
  const F        = babyJub.F;

  // 2) Your message and randomness (32‑bit each)
  const inputData = JSON.parse(fs.readFileSync("input.json", "utf-8"));
    const messageArray = inputData.messageArray;
    const r = inputData.r;
    const d = inputData.d;
    const idx = inputData.idx;
    const allPoints = inputData.allPoints;

//   const messageArray = [
//     1,1,1,1,1,1,1,1,1,1
//   ]; // example values
//   const n = messageArray.length;
//   const r = 0x12345678;
//   const d = 0.35;
//   const idx = 0;

//   const allPoints = [
//     Array(10).fill(1.1),
//     Array(10).fill(2.0),
//     Array(10).fill(3.0),
//     Array(10).fill(4.0),
//     Array(10).fill(5.0)
//   ]

  // Scaling 
  const scaledMsg = messageArray.map(x => Math.round(x * SCALE));
  const scaledPts = allPoints.map(arr => arr.map(x => Math.round(x * SCALE)));
  const scaledD = Math.round(d * SCALE);

  // 3) Bit‑decompose into little‑endian arrays
  const msgBits  = scaledMsg.map(x => toBitsLE(x, 32));
  const randBits = toBitsLE(r,  32);
  const allBits  = msgBits.flat().concat(randBits);

  // 4) Pack 8 bits per byte, LE
  const inputBuffer = packBitsLE(allBits);

  // 5) Hash to a BabyJub point
  //    pedersen.hash(buf) consumes packed bytes exactly as Circom does
  const hashPoint = pedersen.hash(inputBuffer);

  // 6) Unpack to affine coordinates [x, y]
  const P = babyJub.unpackPoint(hashPoint);
  const commitX = F.toObject(P[0]);
  const commitY = F.toObject(P[1]);

  console.log("commitX =", commitX);
  console.log("commitY =", commitY);

  // 7) Emit the snarkjs input.json
const input = {
    commitX: commitX.toString(),
    commitY: commitY.toString(),
    arrBits: msgBits,
    randBits: randBits,
    points: scaledPts,
    d: scaledD,
    idx: idx,
};
  fs.writeFileSync("input.json", JSON.stringify(input, null, 2));
}

main().catch(console.error);
