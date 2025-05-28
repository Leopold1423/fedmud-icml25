import os
import sys
import copy
import time
import torch
import argparse
import warnings

sys.path.append(".")
sys.path.append("..")
warnings.filterwarnings("ignore")
import flwr as fl
from collections import OrderedDict
from train.trainer import train
from utils.logger import get_log
from utils.tool import get_device
from dataloader.dataloader import get_data, get_client_train_dataloader
from decompose.models import get_model_with_config_force
from decompose.dmu import dmu_push_reset_update


class Client(fl.client.NumPyClient):
    def __init__(self, args: argparse.Namespace):
        self.device = get_device(args.gpu)
        self.pt_path = os.path.join(args.log_dir, "results")
        self.logger = get_log(self.pt_path, "client-" + str(args.id))
        self.logger.info(args)
        self.id, self.com_type, self.past_parameters = args.id, args.com_type, None
        self.model_name, self.dataset_name = args.model, args.dataset
        self.part_strategy, self.num_client, self.val_ratio = (
            args.part_strategy,
            args.num_client,
            args.val_ratio,
        )
        self.train_data = get_data(self.dataset_name)

        model_config = {
            "com_type": args.com_type,
            "ratio": args.ratio,
            "init_type_mag": args.init_type_mag,
            "skip_front_back": args.skip_front_back,
            "dmu_type": args.dmu_type,
            "dmu_pattern": args.dmu_pattern,
        }
        self.model, ratio = get_model_with_config_force(
            self.model_name, self.dataset_name, model_config
        )
        self.model.to(self.device)
        self.logger.info(model_config)

        self.keys_weight = []
        for k, val in self.model.named_modules():
            if isinstance(val, (torch.nn.Conv2d, torch.nn.Linear)):
                self.keys_weight.append(k + ".weight")

    def set_parameters(self, parameters, config):
        if self.com_type in ["fedavg"]:
            params_zip = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_zip})
            self.past_parameters = copy.deepcopy(state_dict)
            self.model.load_state_dict(state_dict, strict=True)

        if self.com_type == "feddmu":
            params_zip = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_zip})
            self.past_parameters = copy.deepcopy(state_dict)
            self.model.load_state_dict(state_dict, strict=True)
            dmu_push_reset_update(self.model, config)  # dmu, push update

    def get_parameters(self, config):
        state_dict = copy.deepcopy(self.model.state_dict())
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def fit(self, parameters, config):
        config["ids"] = [int(num) for num in config["ids"].split(".")]
        self.set_parameters(parameters, config)
        trainloader, valloader = get_client_train_dataloader(
            self.train_data,
            self.dataset_name,
            self.part_strategy,
            self.num_client,
            config["ids"][self.id],
            config["batch_size"],
            self.val_ratio,
        )
        results = train(self.model, trainloader, valloader, config, self.device)
        self.logger.info(
            "round %d client #%d, val loss: %.4f, val acc: %.4f"
            % (
                config["round"],
                config["ids"][self.id],
                results["val_loss"],
                results["val_accuracy"],
            )
        )
        parameters_prime = self.get_parameters(config)
        return parameters_prime, len(trainloader), results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="client")
    parser.add_argument("--ratio", type=float, default=0.1, help="compress ratio")
    parser.add_argument(
        "--init_type_mag", type=str, default="uni_0.3", help="init_type_mag"
    )
    parser.add_argument(
        "--skip_front_back", type=str, default="1-1", help="skip front layer number"
    )
    parser.add_argument(
        "--dmu_type", type=str, default="low", help="decomposition type"
    )  # feddmu
    parser.add_argument(
        "--dmu_pattern", type=str, default="ab", help="ab/fab/fab+cfd"
    )  # feddmu

    parser.add_argument("--com_type", type=str, default="fedavg", help="type")
    parser.add_argument("--model", type=str, default="c4l1", help="model")
    parser.add_argument("--dataset", type=str, default="fmnist", help="dataset")
    parser.add_argument("--part_strategy", type=str, default="iid", help="iid")
    parser.add_argument(
        "--num_client", type=int, default=30, choices=range(2, 400), help="num_client"
    )
    parser.add_argument(
        "--id", type=int, default=9, choices=range(0, 400), help="client id"
    )
    parser.add_argument("--val_ratio", type=float, default=0.1, help="dataset")
    parser.add_argument("--gpu", type=int, default=4, help="-1 0 1")
    parser.add_argument(
        "--ip", type=str, default="0.0.0.0:12345", help="server address"
    )
    parser.add_argument("--log_dir", type=str, default="./log/debug/", help="dir")
    args = parser.parse_args()
    client = Client(args)
    while True:  # wait for server init
        flags_path = os.path.join(client.pt_path, "flags_" + str(args.id) + ".npy")
        if os.path.exists(flags_path):
            os.remove(flags_path)
            time.sleep(1)
            break
        else:
            time.sleep(1)
    print("start client {}".format(args.id))
    fl.client.start_numpy_client(server_address=args.ip, client=client)
