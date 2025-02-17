import torch
import torch.nn as nn
import torchvision.models as models

def get_model(model):
    if model == "resnet18":
        outmodel = models.resnet18(pretrained=True)
    elif model == "vgg16":
        outmodel = models.vgg16(pretrained=True)
    else:
        raise ValueError("Unknown model")

    if torch.cuda.is_available():
        outmodel = outmodel.cuda()
    
    return outmodel