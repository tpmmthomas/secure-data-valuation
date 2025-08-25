import os
import random
import torch
import torch.nn as nn
import ezkl
import numpy as np
import json
import time

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

def get_all_modules(module):
    """Recursively get all modules in order."""
    modules = []
    for child in module.children():
        if isinstance(child, nn.Sequential):
            # Flatten Sequential containers
            modules.extend(get_all_modules(child))
        elif len(list(child.children())) == 0:
            # Leaf module
            modules.append(child)
        else:
            # Intermediate module with children
            modules.extend(get_all_modules(child))
    return modules

def choose_random_layers(model, num_layers):
    layer_num = len(get_all_modules(model))
    return random.sample(list(range(layer_num)), num_layers)

def get_layer(model, layer_idx):
    all_layers = get_all_modules(model)
    return all_layers[layer_idx]

def collect_sequential_activations(
    model: nn.Sequential,
    x: torch.Tensor,
    eval_mode: bool = True,   # set False if you want training-time behavior (e.g., Dropout)
    detach: bool = True,      # set False if you want to keep the graph for backprop
    to_cpu: bool = False      # set True if you want smaller GPU memory footprint
):
    """
    Run a batch through a Sequential model and capture each layer's output.

    Returns:
        OrderedDict mapping "idx:LayerClass" -> activation tensor.
        The last item is the model's final output.
    """
    if not isinstance(model, nn.Sequential):
        raise TypeError("model must be an instance of torch.nn.Sequential")
    if eval_mode:
        model.eval()

    activations = dict()
    activations[-1] = x.cpu() if to_cpu else x
    out = x
    for idx, layer in enumerate(model):
        out = layer(out)
        t = out
        if detach:
            t = t.detach()
        if to_cpu:
            t = t.cpu()
        activations[idx] = t

    return activations

async def setup_zkp(model, test_data, layer, mode="pw"):
    '''
        mode= 'pw' or 'pi'
        pw means public weights
        pi means public inputs
    '''
    l = get_layer(model,layer)
    activations = collect_sequential_activations(model,test_data)
    x = activations[layer-1]
    
    # Create layer-specific file paths
    layer_model_path = os.path.join('data', f'layer_{layer}_network.onnx')
    layer_compiled_path = os.path.join('data', f'layer_{layer}_network.compiled')
    layer_pk_path = os.path.join('data', f'layer_{layer}_test.pk')
    layer_vk_path = os.path.join('data', f'layer_{layer}_test.vk')
    layer_settings_path = os.path.join('data', f'layer_{layer}_settings.json')
    
    torch.onnx.export(l,               # model being run
        x,                   # model input (or a tuple for multiple inputs)
        layer_model_path,            # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=10,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    'output' : {0 : 'batch_size'}})

    py_run_args = ezkl.PyRunArgs()
    if mode == 'pw':
        py_run_args.input_visibility = "private" 
        py_run_args.output_visibility = "public"
        py_run_args.param_visibility = "fixed" 
    elif mode == 'pi':
        py_run_args.input_visibility = "public" #Bob can see this
        py_run_args.output_visibility = "hashed" #This hash is given to Bob
        py_run_args.param_visibility = "private" 
    else:
        raise NotImplementedError("Not implemented")

    res = ezkl.gen_settings(layer_model_path, layer_settings_path, py_run_args=py_run_args)
    assert res

    # cal_images = np.array([trainset[i][0].numpy() for i in indices])

    # #Alice should use some real data to calibrate the model, here we use random data
    # data_array = (cal_images).reshape([-1]).tolist()

    # data = dict(input_data = [data_array])

    # # Serialize data into file:
    # json.dump(data, open(cal_path, 'w'))

    # await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")
    res = ezkl.compile_circuit(layer_model_path, layer_compiled_path, layer_settings_path)
    assert res 

    # srs path - This actually requires a trusted setup.
    res = await ezkl.get_srs(layer_settings_path)
    res = ezkl.setup(
            layer_compiled_path,
            layer_vk_path,
            layer_pk_path,
        )

    assert res
    assert os.path.isfile(layer_vk_path)
    assert os.path.isfile(layer_pk_path)
    assert os.path.isfile(layer_settings_path)
        
async def prove_zkp(model, data, layer):
    input_data = data
    activations = collect_sequential_activations(model,input_data)
    input_data = activations[layer-1]
    output_data = activations[layer]

    # Create layer-specific file paths
    layer_compiled_path = os.path.join('data', f'layer_{layer}_network.compiled')
    layer_pk_path = os.path.join('data', f'layer_{layer}_test.pk')
    layer_vk_path = os.path.join('data', f'layer_{layer}_test.vk')
    layer_settings_path = os.path.join('data', f'layer_{layer}_settings.json')
    layer_witness_path = os.path.join('data', f'layer_{layer}_witness.json')
    layer_data_path = os.path.join('data', f'layer_{layer}_input.json')
    layer_output_path = os.path.join('data', f'layer_{layer}_output.json')
    layer_proof_path = os.path.join('data', f'layer_{layer}_test.pf')

    # for pts in points_to_submit:
    data_array = [img.reshape(-1).tolist() for img in input_data]
    data = dict(input_data = data_array)
    json.dump(data, open(layer_data_path, 'w'))

    #Save the output
    data = dict(output_data = output_data.cpu().numpy().tolist())
    json.dump(data, open(layer_output_path, 'w'))

    #Generate witness
    res = await ezkl.gen_witness(layer_data_path, layer_compiled_path, layer_witness_path)
    assert res
    time.sleep(0.1)
    #Prove
    res = ezkl.prove(layer_witness_path, layer_compiled_path, layer_pk_path, layer_proof_path, "single")
    assert res
    
async def verify_zkp(layer):
    layer_proof_path = os.path.join('data', f'layer_{layer}_test.pf')
    layer_vk_path = os.path.join('data', f'layer_{layer}_test.vk')
    layer_settings_path = os.path.join('data', f'layer_{layer}_settings.json')
    
    res = ezkl.verify(layer_proof_path, layer_settings_path, layer_vk_path)
    assert res