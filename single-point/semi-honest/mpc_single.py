#!/usr/bin/env python3

# pyre-unsafe

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import shutil
import tempfile
import time

import crypten
import crypten.communicator as comm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from examples.meters import AverageMeter
from examples.util import NoopContextManager
from torchvision import datasets, transforms

from model import Model, INPUT_SIZE



def run_data_valuation(
    seed=None,
    context_manager=None,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    crypten.init()

    lossFunc = crypten.nn.CrossEntropyLoss()

    if context_manager is None:
        context_manager = NoopContextManager()

    data_dir = tempfile.TemporaryDirectory()
    
    #Load model to Alice
    private_model = construct_private_model(INPUT_SIZE)
    private_input, private_label = construct_private_data()
    
    private_model.eval()
    output = private_model(private_input)
    loss_value = -lossFunc(output, private_label)
    
    logging.info("Loss value: %s" % loss_value.get_plain_text())
    
    data_dir.cleanup()

def construct_private_model(input_size):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size)

    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = Model()
        model_upd.load_state_dict(torch.load('./data/model.pth', weights_only=True))
    else:
        model_upd = Model()
    private_model = crypten.nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    return private_model

def construct_private_data():
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()

    if world_size > 1:
        # party 1 gets the actual tensor; remaining parties get dummy tensor
        src_id = 1
    else:
        # party 0 gets the actual tensor since world size is 1
        src_id = 0

    if rank == src_id:
        input_upd = torch.load('data/data.pth', weights_only=True)
        label_upd = torch.load('data/lbl.pth', weights_only=True)
    else:
        input_upd = torch.empty(INPUT_SIZE)
        label_upd = torch.empty((10,))
    private_input = crypten.cryptensor(input_upd, src=src_id)
    private_label = crypten.cryptensor(label_upd, src=src_id)
    return private_input, private_label
