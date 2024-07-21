# standard libraries
import argparse
import yaml
from pathlib import Path
import time
import matplotlib.pyplot as plt

#import torch related libraries
import torch
from torchvision.utils import save_image


# load modules from local files
from AMLsrc.utilities.modules import recursive_find_python_class
from AMLsrc.data.dataloader import get_MNIST_dataloader
from AMLsrc.utilities.trainer import train_flow, train
from AMLsrc.utilities.metrics import get_eval_metrics


print("Setting up configuration")
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--version", type=str, default="test_version")
args = parser.parse_args()


# add content of config file to args
with open(args.config, "r") as stream:
    config = yaml.safe_load(stream)
    for key in config:
        setattr(args, key, config[key])

#add version to name
args.name = args.name + "_" + args.version

# print all arguments
for key, value in sorted(vars(args).items()):
    print(key, "=", value)

# Set device
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {args.device}")

# Set random seed
print(f"Using random seed: {args.seed}")
torch.manual_seed(args.seed)


# Load data
print("Loading data")
mnist_train_loader, mnist_test_loader = get_MNIST_dataloader(batch_size=args.batch_size, transform_description=args.transform_description)
print(f"Size of train data: {len(mnist_train_loader.dataset)}")
print(f"Size of test data: {len(mnist_test_loader.dataset)}")



# load moduels and define model
print(f"Setting up model with latent dimension: {args.dim}")
if args.model == "VAE":
    encoder_net, decoder_net = recursive_find_python_class(args.encoder_decoder_nets)(args.latent_dim,unflatten=False)()
    encoder = recursive_find_python_class(args.encoder)(encoder_net)
    decoder = recursive_find_python_class(args.decoder)(decoder_net)
    prior = recursive_find_python_class(args.prior)(args.latent_dim, encoder=encoder)
    model = recursive_find_python_class(args.model)(prior, decoder, encoder).to(args.device)
    train = train
else:
    model = recursive_find_python_class(args.model)(args).to(args.device)
    train = train_flow



print("Training model")
print(f"Using Adam optimizer with learning rate: {args.lr}")
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train(model, optimizer, mnist_train_loader, args.epochs, args.device)

# # Save model
output_dir = Path("models").joinpath(args.name).joinpath(str(int(time.time())))
output_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), output_dir.joinpath("model.pt"))
with open(output_dir.joinpath("config.yaml"), "w") as stream:
    yaml.dump(vars(args), stream)


# Sample from model
print("Sampling from model")
model.eval()
model.to('cpu')
with torch.no_grad():
    #sample p(x|z) for 64 samples
    samples = (model.sample(64)).cpu()
    save_image(samples.view(64, 1, 28, 28), output_dir / "samples.png", nrow=8)

