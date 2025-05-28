import sys

sys.path.append(".")
sys.path.append("..")
import torch.nn as nn

from decompose.dmu import dmu_replace_modules
from model.cnn import CNN, CNN_CIFAR, CNN_MNIST


def count_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)


def get_model(model_name, dataset):
    if dataset == "cifar100":
        if "cnn" in model_name:
            model = CNN_CIFAR(100)
    elif dataset == "cifar10":
        if "cnn" in model_name:
            model = CNN_CIFAR()
    elif dataset == "fmnist" or dataset == "mnist":
        if "cnn" in model_name:
            model = CNN_MNIST()
    elif dataset == "svhn":
        if "cnn" in model_name:
            model = CNN()
    else:
        raise ValueError("wrong dataset.")

    weight_init(model)
    return model


def get_skip_layers(model, front_back):
    front, back = int(front_back.split("-")[0]), int(front_back.split("-")[1])
    layers = []
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append(name)
            layer_idx += 1
    return layers[:front] + layers[::-1][:back]


def get_model_with_config(model_name, dataset, config=None):
    model = get_model(model_name, dataset)
    num_para = count_trainable_parameters(model)

    skip_layers = get_skip_layers(model, config["skip_front_back"])

    if "dmu" in config["com_type"]:
        dmu_replace_modules(model, config, skip_layers)

    num_para_compress = count_trainable_parameters(model)
    real_ratio = num_para_compress / num_para
    return model, real_ratio


def get_model_with_config_force(model_name, dataset, config=None):
    model, real_ratio = get_model_with_config(model_name, dataset, config)
    target_ratio = config["ratio"]

    if config["com_type"] in ["feddmu"]:
        while real_ratio - target_ratio > 0.001:
            config["ratio"] -= 0.001
            if config["ratio"] <= 0:
                break
            model, real_ratio = get_model_with_config(model_name, dataset, config)

    return model, real_ratio
