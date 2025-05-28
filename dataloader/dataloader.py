import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100

workspace = "your_path/feddmu"


def get_data(dataset):
    if dataset == "mnist":
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train = MNIST(
            root=workspace + "/dataset/",
            train=True,
            download=True,
            transform=train_transform,
        )
    if dataset == "fmnist":
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        )
        train = FashionMNIST(
            root=workspace + "/dataset/",
            train=True,
            download=True,
            transform=train_transform,
        )
    if dataset == "svhn":
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4496,), (0.1995,))]
        )
        train = SVHN(
            root=workspace + "/dataset/",
            split="train",
            download=True,
            transform=train_transform,
        )
    if dataset == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        train = CIFAR10(
            root=workspace + "/dataset/",
            train=True,
            download=True,
            transform=train_transform,
        )
    if dataset == "cifar100":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]
                ),
            ]
        )
        train = CIFAR100(
            root=workspace + "/dataset/",
            train=True,
            download=True,
            transform=train_transform,
        )
    return train


def get_client_train_dataloader(
    train,
    dataset,
    part_strategy,
    num_clients,
    id=0,
    batch_size=64,
    val_ratio=0.0,
    seed=1234,
):
    # train = get_data(dataset)

    npy_name = (
        dataset
        + "-"
        + part_strategy
        + "-"
        + str(num_clients)
        + "-"
        + str(seed)
        + ".npy"
    )
    train_id_map = np.load(
        workspace + "/dataloader/_npy_/" + npy_name, allow_pickle=True
    )
    train_id_map = train_id_map.item()

    client_data = torch.utils.data.Subset(train, train_id_map[id])
    n_valset = int(len(client_data) * val_ratio)
    valset = torch.utils.data.Subset(client_data, range(0, n_valset))
    trainset = torch.utils.data.Subset(client_data, range(n_valset, len(client_data)))
    valLoader = DataLoader(valset, batch_size=batch_size)
    trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainLoader, valLoader


def get_server_test_dataloader(dataset, batch_size):
    if dataset == "mnist":
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1326,), (0.3106,))]
        )
        test = MNIST(
            root=workspace + "/dataset/",
            train=False,
            download=True,
            transform=test_transform,
        )
    if dataset == "fmnist":
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2868,), (0.3524,))]
        )
        test = FashionMNIST(
            root=workspace + "/dataset/",
            train=False,
            download=True,
            transform=test_transform,
        )
    if dataset == "svhn":
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4560,), (0.2244,))]
        )
        test = SVHN(
            root=workspace + "/dataset/",
            split="test",
            download=True,
            transform=test_transform,
        )
    if dataset == "cifar10":
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        test = CIFAR10(
            root=workspace + "/dataset/",
            train=False,
            download=True,
            transform=test_transform,
        )
    if dataset == "cifar100":
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]
                ),
            ]
        )
        test = CIFAR100(
            root=workspace + "/dataset/",
            train=False,
            download=True,
            transform=test_transform,
        )

    test_Loader = DataLoader(test, batch_size=batch_size)
    return test_Loader


if __name__ == "__main__":
    print("done")
