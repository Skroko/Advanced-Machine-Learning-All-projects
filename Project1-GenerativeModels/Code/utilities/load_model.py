import argparse
import yaml
from pathlib import Path
import time
import matplotlib.pyplot as plt

#import torch related libraries
import torch
from torchvision.utils import save_image
import yaml


# load modules from local files
from AMLsrc.utilities.modules import recursive_find_python_class
class Args:
    pass

def load_model(path: Path, device: str = 'cpu'):
    # load model
    model_path = path / "model.pt"
    config_path = path / "config.yaml"
    state_dict = torch.load(model_path, map_location=device)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    args = Args()
    for key in config:
        setattr(args, key, config[key])
    args.device = device
    
    encoder_net, decoder_net = recursive_find_python_class(args.encoder_decoder_nets)(args.latent_dim)()
    encoder = recursive_find_python_class(args.encoder)(encoder_net)
    decoder = recursive_find_python_class(args.decoder)(decoder_net)
    prior = recursive_find_python_class(args.prior)(args.latent_dim, encoder=encoder)
    model = recursive_find_python_class(args.model)(prior, decoder, encoder).to(args.device)

    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_flow_model(path: Path, device: str = 'cpu'):
    # load model
    model_path = path / "model.pt"
    config_path = path / "config.yaml"
    state_dict = torch.load(model_path, map_location=device)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    args = Args()
    for key in config:
        setattr(args, key, config[key])
    args.device = device
    
    model = recursive_find_python_class(args.model)(args).to(args.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
