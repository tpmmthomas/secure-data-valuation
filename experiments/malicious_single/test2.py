import os
import importlib
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import ezkl
import os
import torchvision
import json
import random 
import numpy as np
import torchvision.transforms as transforms
from launcher import run_experiment
import onnx
import threading, time, psutil, os
import asyncio


def watch_mem():
    p = psutil.Process(os.getpid())
    while True:
        print("MEM:", p.memory_info().rss // (1024**2), "MB")
        time.sleep(5)

#Specifying some path parameters
model_path = os.path.join('data','network.onnx')
compiled_model_path = os.path.join('data','network.compiled')
pk_path = os.path.join('data','test.pk')
vk_path = os.path.join('data','test.vk')
settings_path = os.path.join('data','settings.json')
cal_path = os.path.join('data',"calibration.json")
witness_path = os.path.join('data','witness.json')
data_path = os.path.join('data','input.json')
output_path = os.path.join('data','output.json')
label_path = os.path.join('data','label.json')
proof_path = os.path.join('data','test.pf')

# List of models to benchmark.
models = ["ResNet18"] # 

with open("results/exp4.txt", "w") as f:
        pass
    
def fprint(msg,fn="results/exp4.txt"):
    print(msg)
    with open(fn, "a") as f:
        f.write(msg + "\n")
        
# Set torch seeds
torch.manual_seed(42)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True,transform=transform)
indices = random.sample(range(len(trainset)), 3)
cal_images = np.array([trainset[i][0].numpy() for i in indices])
data_array = (cal_images).reshape([-1]).tolist()

data = dict(input_data = [data_array])

async def setup(i, run_args):
    # file names
    model_path = os.path.join('data','network_split_'+str(i)+'.onnx')
    settings_path = os.path.join('data','settings_split_'+str(i)+'.json')
    data_path =  os.path.join('data','input_'+str(i)+'.json')
    compiled_model_path = os.path.join('data','network_split_'+str(i)+'.compiled')
    pk_path = os.path.join('data','test_split_'+str(i)+'.pk')
    vk_path = os.path.join('data','test_split_'+str(i)+'.vk')
    witness_path = os.path.join('data','witness_split_'+str(i)+'.json')

    if i > 0:
         prev_witness_path = os.path.join('data','witness_split_'+str(i-1)+'.json')
         witness = json.load(open(prev_witness_path, 'r'))
         data = dict(input_data = witness['outputs'])
         # Serialize data into file:
         json.dump(data, open(data_path, 'w' ))
    else:
         data_path = os.path.join('data','input_0.json')

    # generate settings for the current model
    res = ezkl.gen_settings(model_path, settings_path, py_run_args=run_args)
    res = await ezkl.calibrate_settings(data_path, model_path, settings_path, "resources", scales=[run_args.input_scale], max_logrows=run_args.logrows)
    assert res == True

    # load settings and print them to the console
    settings = json.load(open(settings_path, 'r'))
    settings['run_args']['logrows'] = run_args.logrows
    json.dump(settings, open(settings_path, 'w' ))

    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)

    print("Running setup")
    res = ezkl.setup(
         compiled_model_path,
         vk_path,
         pk_path,
      )
    print("Successful setup")

    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)

    res = await ezkl.gen_witness(data_path, compiled_model_path, witness_path, vk_path)
    run_args.input_scale = settings["model_output_scales"][0]
    return run_args

# Serialize data into file:
json.dump(data, open(cal_path, 'w'))
async def main():
    for model_name in models:
        total_times = []
        total_memory = []
        total_precision = []
        total_comm = []
        # Create the model and save it in the data directory.
        model_module = importlib.import_module("model")
        ModelClass = getattr(model_module, model_name)
        model = ModelClass()
        model.eval()
        torch.save(model.state_dict(), "data/model.pth")
        #Setup the model
        data = torch.load("data/data.pth")
        torch.onnx.export(model,               # model being run
                        data,                   # model input (or a tuple for multiple inputs)
                        model_path,            # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=10,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
        
        #Save the data with the split
        data_path_0 = os.path.join(os.path.dirname(data_path), "input_0.json")
        data_t = dict(input_data = [((data).detach().numpy()).reshape([-1]).tolist()])
        json.dump( data_t, open(data_path_0, 'w' ))

        inter_1 = model.split_1(data)
        data_path_1 = os.path.join(os.path.dirname(data_path), "input_1.json")
        data_t = dict(input_data = [((inter_1).detach().numpy()).reshape([-1]).tolist()])
        json.dump( data_t, open(data_path_1, 'w' ))

        inter_2 = model.split_2(data)
        data_path_2 = os.path.join(os.path.dirname(data_path), "input_2.json")
        data_t = dict(input_data = [((inter_2).detach().numpy()).reshape([-1]).tolist()])
        json.dump( data_t, open(data_path_2, 'w' ))

        inter_3 = model.split_3(data)
        data_path_3 = os.path.join(os.path.dirname(data_path), "input_3.json")
        data_t = dict(input_data = [((inter_3).detach().numpy()).reshape([-1]).tolist()])
        json.dump( data_t, open(data_path_3, 'w' ))

        inter_4 = model.split_4(data)
        data_path_4 = os.path.join(os.path.dirname(data_path), "input_4.json")
        data_t = dict(input_data = [((inter_4).detach().numpy()).reshape([-1]).tolist()])
        json.dump( data_t, open(data_path_4, 'w' ))

        inter_5 = model.split_5(data)
        data_path_5 = os.path.join(os.path.dirname(data_path), "input_5.json")
        data_t = dict(input_data = [((inter_5).detach().numpy()).reshape([-1]).tolist()])
        json.dump( data_t, open(data_path_5, 'w' ))
        

        # model = onnx.load("data/network.onnx")
        # for node in model.graph.node:
        #     print(node.output)
        # exit()

        #Split model
        model_path_0 = os.path.join(os.path.dirname(model_path), "network_split_0.onnx")
        input_names=["input"]
        output_names = ['/relu/Relu_output_0'] #['/layer1/layer1.1/relu_1/Relu_output_0']
        onnx.utils.extract_model(model_path, model_path_0, input_names, output_names)

        model_path_1 = os.path.join(os.path.dirname(model_path), "network_split_1.onnx")
        input_names= ['/relu/Relu_output_0'] #['/layer1/layer1.1/relu_1/Relu_output_0']
        output_names = ['/layer1/layer1.1/relu_1/Relu_output_0']
        onnx.utils.extract_model(model_path, model_path_1, input_names, output_names)

        model_path_2 = os.path.join(os.path.dirname(model_path), "network_split_2.onnx")
        input_names= ['/layer1/layer1.1/relu_1/Relu_output_0'] #['/layer1/layer1.1/relu_1/Relu_output_0']
        output_names = ['/layer2/layer2.1/relu_1/Relu_output_0']
        onnx.utils.extract_model(model_path, model_path_2, input_names, output_names)

        model_path_3 = os.path.join(os.path.dirname(model_path), "network_split_3.onnx")
        input_names= ['/layer2/layer2.1/relu_1/Relu_output_0'] #['/layer1/layer1.1/relu_1/Relu_output_0']
        output_names = ['/layer3/layer3.1/relu_1/Relu_output_0']
        onnx.utils.extract_model(model_path, model_path_3, input_names, output_names)

        model_path_4 = os.path.join(os.path.dirname(model_path), "network_split_4.onnx")
        input_names= ['/layer3/layer3.1/relu_1/Relu_output_0'] #['/layer1/layer1.1/relu_1/Relu_output_0']
        output_names = ['/layer4/layer4.1/relu_1/Relu_output_0']
        onnx.utils.extract_model(model_path, model_path_4, input_names, output_names)

        model_path_5 = os.path.join(os.path.dirname(model_path), "network_split_5.onnx")
        input_names= ['/layer4/layer4.1/relu_1/Relu_output_0'] #['/layer1/layer1.1/relu_1/Relu_output_0']
        output_names = ['output']
        onnx.utils.extract_model(model_path, model_path_5, input_names, output_names)
        
        run_args = ezkl.PyRunArgs()
        run_args.input_visibility = "public"
        run_args.param_visibility = "private"
        run_args.output_visibility = "polycommit"
        run_args.variables = [("batch_size", 1)]
        run_args.input_scale = 2
        run_args.logrows = 20
        run_args.num_inner_cols = 3

        ezkl.get_srs(logrows=run_args.logrows, commitment=ezkl.PyCommitments.KZG)

        new_run_args = await setup(0, run_args)
        for i in range(1, 5):
            new_run_args.input_visibility = "polycommit"
            new_run_args.output_visibility = "polycommit"
            new_run_args = await setup(i, new_run_args)
        new_run_args.output_visibility = "public"
        _= await setup(5, new_run_args)
        print("OK till here")

    # split_io_names = [
    #     (["input"], ["/relu/Relu_output_0"]),
    #     (["/relu/Relu_output_0"], ["/layer1/layer1_output"]),
    #     (["/layer1/layer1_output"], ["/layer2/layer2_output"]),
    #     (["/layer2/layer2_output"], ["/layer3/layer3_output"]),
    #     (["/layer3/layer3_output"], ["/layer4/layer4_output"]),
    #     (["/layer4/layer4_output"], ["output"])
    # ]

    # for i, (inputs, outputs) in enumerate(split_io_names):
    #     split_model_path = os.path.join(os.path.dirname(model_path), f"network_split_{i}.onnx")
    #     onnx.utils.extract_model(model_path, split_model_path, inputs, outputs)

    # new_run_args = run_args
    # for i in range(6):
    #     new_run_args = await setup(i, new_run_args)


#         py_run_args = ezkl.PyRunArgs()
#         py_run_args.input_visibility = "public" #Bob can see this
#         py_run_args.output_visibility = "hashed/public" #This hash is given to Bob
#         py_run_args.param_visibility = "private" 
#         py_run_args.logrows = 14
#         print("Generating settings")
#         res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
#         assert res
#         # await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")
#         print("Compiling circuit")
#         res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
#         assert res
#         # print("Generating srs")
#         res = await ezkl.get_srs(settings_path)
#         print("Setup here")
#         res = ezkl.setup(
#             compiled_model_path,
#             vk_path,
#             pk_path,
#         )
#         print("Done!")


t = threading.Thread(target=watch_mem, daemon=True)
t.start()
asyncio.run(main())