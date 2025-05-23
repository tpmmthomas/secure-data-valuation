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
import importlib

from model import INPUT_SIZE



def run_data_valuation(
    seed=None,
    model_name="LeNet",
    context_manager=None,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    crypten.init()

    lossFunc = crypten.nn.loss.CrossEntropyLoss()

    if context_manager is None:
        context_manager = NoopContextManager()

    data_dir = tempfile.TemporaryDirectory()
    
    #Load model to Alice
    private_model = construct_private_model(INPUT_SIZE, model_name)
    private_input, private_label = construct_private_data()
    
    private_model.eval()
    output = private_model(private_input)
    loss_value = lossFunc(output, private_label)
    
    print("Output: %s" % output.get_plain_text())
    print("Loss value: %s" % loss_value.get_plain_text())
    rank = comm.get().get_rank()
    print(f"Rank: {rank} Communication {comm.get().get_communication_stats()}" )
    
    #Output to a file
    # rank = comm.get().get_rank()
    # if rank == 0:
    #     with open("results/data_valuation.txt", "w") as f:
    #         f.write(str(loss_value.get_plain_text()))
    
    data_dir.cleanup()

def construct_private_model(input_size, model_name):
    
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size)
    
    model_module = importlib.import_module("model")
    ModelClass = getattr(model_module, model_name)


    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = ModelClass()
        model_upd.load_state_dict(torch.load('./data/model.pth'))
    else:
        model_upd = ModelClass()
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
        input_upd = torch.load('data/data.pth')
        label_upd = torch.load('data/lbl.pth')
    else:
        input_upd = torch.empty(INPUT_SIZE)
        label_upd = torch.empty((1,10))
    private_input = crypten.cryptensor(input_upd, src=src_id)
    private_label = crypten.cryptensor(label_upd, src=src_id)
    return private_input, private_label
