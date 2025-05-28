import os
import sys

sys.path.append("..")
import torch
import random
import numpy as np
from torchvision.datasets import MNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100

workspace = "your_path/feddmu"


def get_data(dataset):
    if dataset == "mnist":
        train = MNIST(root=workspace + "/dataset/", train=True, download=True)
    if dataset == "fmnist":
        train = FashionMNIST(root=workspace + "/dataset/", train=True, download=True)
    if dataset == "svhn":
        train = SVHN(root=workspace + "/dataset/", split="train", download=True)
    if dataset == "cifar10":
        train = CIFAR10(root=workspace + "/dataset/", train=True, download=True)
    if dataset == "cifar100":
        train = CIFAR100(root=workspace + "/dataset/", train=True, download=True)
    return train


def print_client_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    print("data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


def partition_train_data(dataset, y, part_strategy, num_clients, seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_samples, num_labels = y.shape[0], 10
    if dataset == "cifar100":
        num_labels = 100

    if part_strategy == "iid":
        idxs = np.random.permutation(num_samples)
        batch_idxs = np.array_split(idxs, num_clients)
        net_dataidx_map = {i: batch_idxs[i].tolist() for i in range(num_clients)}

    elif "labeldir" in part_strategy:
        min_size = 0
        min_require_size = 10
        net_dataidx_map = {}
        beta = eval(part_strategy[8:])
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(num_labels):
                idx_k = np.where(y == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                proportions = np.array(
                    [
                        p * (len(idx_j) < num_samples / num_clients)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif "labelcnt" in part_strategy:
        num = int(eval(part_strategy[8:]) * num_labels)

        times = np.zeros(num_labels, dtype=np.intc)
        contain = np.zeros((num_clients, num), dtype=np.intc)
        for i in range(num_clients):
            usage_per_image = times / (i + 1)
            contain[i] = np.argsort(usage_per_image)[:num]
            times[contain[i]] += 1
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(num_clients)}
        for i in range(num_labels):
            idx_k = np.where(y == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(num_clients):
                if i in contain[j]:
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                    ids += 1
        net_dataidx_map = {k: v.tolist() for k, v in net_dataidx_map.items()}

    print_client_data_stats(y, net_dataidx_map)

    os.makedirs(workspace + "/dataloader/_npy_/", exist_ok=True)
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
    np.save(workspace + "/dataloader/_npy_/" + npy_name, net_dataidx_map)

    # os.makedirs(workspace+'/dataloader/_json_/', exist_ok=True)
    # json_name = dataset+"-"+part_strategy+"-"+str(num_clients)+"-"+str(seed)+".json"
    # with open(workspace+'/dataloader/_json_/'+json_name, 'w+') as f:
    #     json.dump(net_dataidx_map, f, indent=4)

    return net_dataidx_map


def generate_train_npy(dataset, part_strategy, client, seed=1234):
    train = get_data(dataset)
    if "cifar" in dataset:
        partition_train_data(
            dataset, np.array(train.targets), part_strategy, client, seed
        )
    elif "svhn" in dataset:
        partition_train_data(
            dataset, np.array(train.labels), part_strategy, client, seed
        )
    else:
        partition_train_data(
            dataset, train.targets.data.numpy(), part_strategy, client, seed
        )


if __name__ == "__main__":
    seed = 1234
    clients = [100, 200]
    datasets = ["fmnist", "svhn", "cifar10"]
    part_strategies = ["iid", "labeldir0.3", "labelcnt0.3"]
    for dataset in datasets:
        for part_strategy in part_strategies:
            for client in clients:
                generate_train_npy(dataset, part_strategy, client, seed)

    datasets = ["cifar100"]
    part_strategies = ["iid", "labeldir0.1", "labelcnt0.1"]
    for dataset in datasets:
        for part_strategy in part_strategies:
            for client in clients:
                generate_train_npy(dataset, part_strategy, client, seed)
