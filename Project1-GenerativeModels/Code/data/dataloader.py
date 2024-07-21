from torchvision import datasets, transforms
import torch

DATA_PATH = "data/"


def get_MNIST_dataloader(batch_size=32, transform_description="standard"):
    transform = get_transforms(transform_description)
    mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            DATA_PATH,
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            DATA_PATH,
            train=False,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return mnist_train_loader, mnist_test_loader


def get_transforms(description):
    if description == "dequantized":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
                transforms.Lambda(lambda x: x.flatten()),
            ]
        )
    elif description == "binarized":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x > 0.5).float().squeeze()),
            ]
        )
    elif description == "standard":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.squeeze()),
            ]
        )
    elif description == "flatten":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.squeeze()),
                transforms.Lambda(lambda x: x.flatten()),
            ]
        )
    elif description == "minus_one_to_one":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
                transforms.Lambda(lambda x: (x - 0.5) * 2.0),
                transforms.Lambda(lambda x: x.flatten()),
            ]
        )
    else:
        raise ValueError(f"Unknown description: {description}")
