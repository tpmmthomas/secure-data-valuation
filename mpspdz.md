# How to compile and run files

Compiling:

- `./compile.py tutorial` compiles the file `tutorial.mpc` from `Programs/Source` into `Programs/Bytecode/` bytecodes.
Compiling options:  https://mp-spdz.readthedocs.io/en/latest/compilation.html

Running:

- `Scripts/mascot.sh tutorial` runs the compileed tutorial program with inputs stored at `Player-Data/` directory.

# How to write code

- Reference / docs: https://mp-spdz.readthedocs.io/en/latest/Compiler.html

# Load ML models

- `layers = ml.layers_from_torch(net, training_samples.shape, 128, input_via=0)`
