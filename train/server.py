import os
import sys
import copy
import torch
import socket
import argparse
import numpy as np
import random
import warnings
from datetime import datetime

sys.path.append(".")
sys.path.append("..")
warnings.filterwarnings("ignore")
import flwr as fl
from collections import OrderedDict
from flwr.server.strategy import FedAvg
from train.trainer import test
from decompose.models import get_model_with_config_force
from utils.logger import get_log
from utils.tool import get_device, save_npy_record, write_avg_json
from dataloader.dataloader import get_server_test_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="server")
    parser.add_argument("--ratio", type=float, default=0.1, help="compress ratio")
    parser.add_argument(
        "--init_type_mag", type=str, default="uni_0.3", help="init_type_mag"
    )
    parser.add_argument(
        "--skip_front_back", type=str, default="1-1", help="skip front layer number"
    )
    parser.add_argument(
        "--dmu_type", type=str, default="mat", help="mat/kron"
    )  # feddmu
    parser.add_argument(
        "--dmu_pattern", type=str, default="ab", help="ab/fab/fab+cfd"
    )  # feddmu
    parser.add_argument("--dmu_interval", type=int, default=1, help="integer")  # feddmu

    parser.add_argument(
        "--com_type", type=str, default="fedavg", help="communication type"
    )
    parser.add_argument("--model", type=str, default="cnn", help="model")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument("--lr", type=float, default=0.1, help="lr")
    parser.add_argument("--momentum", type=float, default=0, help="momentum")
    parser.add_argument("--l2", type=float, default=0.0, help="l2")
    parser.add_argument("--rounds", type=int, default=100, help="rounds")
    parser.add_argument("--epochs", type=int, default=1, help="epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--save_round", type=int, default=0, help="save_round")
    parser.add_argument(
        "--num_per_round",
        type=int,
        default=10,
        choices=range(2, 400),
        help="num_per_round",
    )
    parser.add_argument(
        "--num_client", type=int, default=100, choices=range(2, 400), help="num_client"
    )
    parser.add_argument("--gpu", type=int, default=4, help="-1 0 1")
    parser.add_argument(
        "--ip", type=str, default="0.0.0.0:12345", help="server address"
    )
    parser.add_argument("--log_dir", type=str, default="./log/debug/", help="dir")
    args = parser.parse_args()

    server_name = socket.gethostname()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = get_log(args.log_dir, f"server_{server_name}_{timestamp}")
    device = get_device(args.gpu)

    logger.info(f"server_name: {server_name}")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    pt_path = os.path.join(args.log_dir, "results")
    os.makedirs(pt_path, exist_ok=True)

    model_config = {
        "com_type": args.com_type,
        "ratio": args.ratio,
        "init_type_mag": args.init_type_mag,
        "skip_front_back": args.skip_front_back,
        "dmu_type": args.dmu_type,
        "dmu_pattern": args.dmu_pattern,
    }
    model, real_ratio = get_model_with_config_force(
        args.model, args.dataset, model_config
    )
    model.to(device)
    logger.info(model_config)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    print(model)
    record = {"accuracy": [], "loss": []}
    com_dict = {
        "ids": "0.1.2.3.4.5.6.7.8.9",
        "seed": 1234,
        "push": False,  # feddmu
    }

    keys_weight = []
    for k, val in model.named_modules():
        if isinstance(val, (torch.nn.Conv2d, torch.nn.Linear)):
            keys_weight.append(k + ".weight")

    def fit_com_dict(com_dict):
        ids = random.sample(list(range(args.num_client)), args.num_per_round)
        com_dict["ids"] = ".".join(map(str, ids))
        com_dict["seed"] = random.randint(a=0, b=2024)

    def fit_config(server_round: int):
        config = {
            "com_type": args.com_type,
            "round": server_round,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "l2": args.l2,
            "ids": com_dict["ids"],
            "seed": com_dict["seed"],
            "push": com_dict["push"],  # feddmu
        }
        return config

    def get_evaluate_fn(model, dataset):
        test_loader = get_server_test_dataloader(dataset, batch_size=args.batch_size)

        def evaluate(server_round, parameters, config):
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            if server_round == 0:
                loss, accuracy = 0.0, 0.0
            else:
                if args.com_type == "feddmu":  # feddmu
                    dmu_interval = int(args.dmu_interval)  # get push for next round
                    if (server_round % dmu_interval) == 0:
                        logger.info("push in the net round")
                        com_dict["push"] = True
                    else:
                        com_dict["push"] = False

                print("starting server evalutation...")
                loss, accuracy = test(model, test_loader, None, device)
                record["accuracy"].append(accuracy)
                record["loss"].append(loss)
                logger.info(
                    "round %d - server test loss:%.4f; acc:%.4f"
                    % (server_round, loss, accuracy)
                )
                if args.save_round:
                    torch.save(
                        model.state_dict(),
                        os.path.join(pt_path, str(server_round) + ".pt"),
                    )
                    if accuracy >= np.max(np.array(record["accuracy"])):
                        torch.save(model.state_dict(), os.path.join(pt_path, "best.pt"))

            fit_com_dict(com_dict)  # prepare ids and seeds for next round
            return loss, {"accuracy": accuracy}

        return evaluate

    strategy = FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_per_round,
        min_evaluate_clients=0,
        min_available_clients=args.num_per_round,
        evaluate_fn=get_evaluate_fn(model, args.dataset),
        on_fit_config_fn=fit_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    for i in range(args.num_per_round):
        np.save(os.path.join(pt_path, "flags_" + str(i) + ".npy"), [])

    fl.server.start_server(
        server_address=args.ip,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    best_round = np.argmax(np.array(record["accuracy"]))
    best_acc = record["accuracy"][best_round]
    best_loss = record["loss"][best_round]
    logger.info(
        "* best round: %d; best acc: %.4f; best loss: %.4f"
        % (best_round, best_acc, best_loss)
    )

    record["best_accuracy"] = best_acc
    record["best_round"] = best_round
    record["ratio"] = real_ratio
    save_npy_record(pt_path, record, name="record")
    write_avg_json(
        args.log_dir, name="record", tags=["best_accuracy", "ratio", "best_round"]
    )
