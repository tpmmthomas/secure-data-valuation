#!/usr/bin/env python3

# pyre-unsafe

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
To run mpc_cifar example in multiprocess mode:

$ python3 examples/mpc_cifar/launcher.py \
    --evaluate \
    --resume path-to-model/model.pth.tar \
    --batch-size 1 \
    --print-freq 1 \
    --skip-plaintext \
    --multiprocess

To run mpc_cifar example on AWS EC2 instances:

$ python3 scripts/aws_launcher.py \
      --ssh_key_file=$HOME/.aws/fair-$USER.pem \
      --instances=i-038dd14b9383b9d79,i-08f057b9c03d4a916 \
      --aux_files=examples/mpc_cifar/mpc_cifar.py,\
path-to-model/model.pth.tar\
      examples/mpc_cifar/launcher.py \
      --evaluate \
      --resume model.pth.tar \
      --batch-size 1 \
      --print-freq 1 \
      --skip-plaintext
"""

import argparse
import logging
import os
import time
import importlib
import psutil

from multiprocess_launcher import (
    MultiProcessLauncher,
)


parser = argparse.ArgumentParser(description="CrypTen Cifar Training")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--model_name",
    type=str,
    default="Model",
    help="Specifies the model name",
)

def _run_experiment(args):
        # only import here to initialize crypten within the subprocesses
        # pyre-fixme[21]: Could not find module `mpc_cifar`.
        from mpc_single import run_data_valuation  # @manual

        # Only Rank 0 will display logs.
        level = logging.INFO
        if "RANK" in os.environ and os.environ["RANK"] != "0":
            level = logging.CRITICAL
        logging.getLogger().setLevel(level)
        run_data_valuation(
            args.seed, args.model_name
        )

def run_experiment(args=None):
    if args is None:
        args = parser.parse_args()
    launcher = MultiProcessLauncher(2, _run_experiment,args)
    launcher.start()
    launcher.join()
    launcher.terminate()
    
if __name__ == "__main__":
    run_experiment()
        
# def main(run_experiment):
#     args = parser.parse_args()
#     if args.multiprocess:
#         launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
#         launcher.start()
#         launcher.join()
#         launcher.terminate()
#     else:
#         run_experiment(args)
        


# if __name__ == "__main__":
#     main(_run_experiment)
