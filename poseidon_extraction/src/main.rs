use halo2_gadgets::poseidon::primitives::{
    self as poseidon, ConstantLength, Spec,
};
use halo2_proofs::halo2curves::bn256::Fr as Fp; // BN254 scalar field
use halo2_proofs::halo2curves::ff::Field;
use maybe_rayon::slice::ParallelSlice;
use maybe_rayon::prelude::ParallelIterator;

mod poseidon_params;

use halo2_proofs::plonk::Error as PlonkError;
use thiserror::Error;

/// Error type for the circuit module
#[derive(Error, Debug)]
pub enum ModuleError {
    /// Halo 2 error
    #[error("[halo2] {0}")]
    Halo2Error(#[from] PlonkError),
    /// Wrong input type for a module
    #[error("wrong input type {0} must be {1}")]
    WrongInputType(String, String),
    /// A constant was not previously assigned
    #[error("constant was not previously assigned")]
    ConstantNotAssigned,
    /// Input length is wrong
    #[error("input length is wrong {0}")]
    InputWrongLength(usize),
}

impl From<ModuleError> for PlonkError {
    fn from(_e: ModuleError) -> PlonkError {
        PlonkError::Synthesis
    }
}

//
// 1. We define our Poseidon parameters exactly
//    as in EZKL's poseidon_params.rs / spec.rs.
//
//  Typically: 
//      WIDTH = 5, RATE = 4
//      FULL_ROUNDS = 8
//      PARTIAL_ROUNDS = 60
//
//  We'll define some sample arrays below. For production, 
//  copy the entire arrays from EZKL's "poseidon_params.rs".
//
// const POSEIDON_WIDTH: usize = 5;
// const POSEIDON_RATE: usize  = 4;

// const FULL_ROUNDS: usize = 8;
// const PARTIAL_ROUNDS: usize = 60;

// // Below is a truncated example. In reality, you need all 340 round constants
// // for t=5, (FULL_ROUNDS + PARTIAL_ROUNDS)=68, and an MDS matrix 5×5=25 elements.
// #[allow(non_snake_case)]
// pub(crate) static ROUND_CONSTANTS: [[Fr; POSEIDON_WIDTH]; FULL_ROUNDS + PARTIAL_ROUNDS] = {
//     // We show just a few random placeholders here.
//     // Copy from ezkl/src/circuit/modules/poseidon/poseidon_params.rs 
//     // for the real array.
//     [
//         // Each of these is 5 BN254 field elements, which is the Poseidon "width".
//         [Fr::from_u128(1), Fr::from_u128(2), Fr::from_u128(3), Fr::from_u128(4), Fr::from_u128(5)],
//         [Fr::from_u128(6), Fr::from_u128(7), Fr::from_u128(8), Fr::from_u128(9), Fr::from_u128(10)],
//         // ... up to 68 rows total ...
//         [Fr::from_u128(0); 5], // placeholder
//         [Fr::from_u128(0); 5], // placeholder
//         // ...
//     ]
// };

// #[allow(non_snake_case)]
// pub(crate) static MDS: [[Fr; POSEIDON_WIDTH]; POSEIDON_WIDTH] = {
//     // 5×5 MDS matrix. We'll use placeholder values below, but in production
//     // you'd copy from the real MDS in ezkl's poseidon_params.
//     [
//         [Fr::from_u128(1), Fr::from_u128(2), Fr::from_u128(3), Fr::from_u128(4), Fr::from_u128(5)],
//         [Fr::from_u128(6), Fr::from_u128(7), Fr::from_u128(8), Fr::from_u128(9), Fr::from_u128(10)],
//         [Fr::from_u128(11), Fr::from_u128(12), Fr::from_u128(13), Fr::from_u128(14), Fr::from_u128(15)],
//         [Fr::from_u128(16), Fr::from_u128(17), Fr::from_u128(18), Fr::from_u128(19), Fr::from_u128(20)],
//         [Fr::from_u128(21), Fr::from_u128(22), Fr::from_u128(23), Fr::from_u128(24), Fr::from_u128(25)],
//     ]
// };

//
// 2. Implement the `Spec` trait.
//    We replicate the same "Pow5 Poseidon" with alpha=5, 
//    the same full rounds, partial rounds, etc.
//
#[derive(Debug)]
pub struct MyPoseidonSpec;

pub const POSEIDON_WIDTH: usize = 2;
/// The number of full SBox rounds
pub const POSEIDON_RATE: usize = 1;

pub const L : usize = 32;

pub(crate) type Mds<Fp, const T: usize> = [[Fp; T]; T];

impl Spec<Fp, POSEIDON_WIDTH, POSEIDON_RATE> for MyPoseidonSpec {
    fn full_rounds() -> usize {
        8
    }
    fn partial_rounds() -> usize {
        56
    }

    // The alpha for "Poseidon over BN254" is usually 5 (the "Pow5" construction).
    fn sbox(val: Fp) -> Fp {
        val.pow_vartime([5])
    }

    fn secure_mds() -> usize {
        unimplemented!()
    }

    fn constants() -> (
        Vec<[Fp; POSEIDON_WIDTH]>,
        Mds<Fp, POSEIDON_WIDTH>,
        Mds<Fp, POSEIDON_WIDTH>,
    ) {
        (
            poseidon_params::ROUND_CONSTANTS[..].to_vec(),
            poseidon_params::MDS,
            poseidon_params::MDS_INV,
        )
    }
}

fn run(message: Vec<Fp>) -> Result<Vec<Vec<Fp>>, ModuleError> {
    let mut hash_inputs = message;

    let mut one_iter = false;
    // do the Tree dance baby
    while hash_inputs.len() > 1 || !one_iter {
        let hashes: Vec<Fp> = hash_inputs
            .par_chunks(L)
            .map(|block| {
                let mut block = block.to_vec();
                let remainder = block.len() % L;

                if remainder != 0 {
                    block.extend(vec![Fp::ZERO; L - remainder].iter());
                }

                let block_len = block.len();

                let message = block
                    .try_into()
                    .map_err(|_| ModuleError::InputWrongLength(block_len))?;

                Ok(poseidon::Hash::
                    <_,
                    MyPoseidonSpec, 
                    ConstantLength<L>, 
                    POSEIDON_WIDTH, 
                    POSEIDON_RATE>
                ::init()
                .hash(message))
            })
            .collect::<Result<Vec<_>, ModuleError>>()?;
        one_iter = true;
        hash_inputs = hashes;
    }

    Ok(vec![hash_inputs])
}

fn main() {
    // 3. Let's do a basic test hashing a vector of 3 elements:
    let inputs = [Fp::from(0)];

    // The library's "Hash" function can be used for a fixed-size input:
    // For dynamic length, you'd do a "ConstantLength<N>" that matches the number of inputs.
    // Because we use RATE=4, we can pass up to 4 elements at once in a single permutation.
    // If your vector is bigger than RATE, you'd need a sponge or "tree" approach.
    let digest = run(inputs.to_vec()).unwrap();

    println!("Poseidon digest = {:?}", digest);
}
