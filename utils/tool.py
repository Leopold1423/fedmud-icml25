import json
import os
import socket
from datetime import datetime

import numpy as np
import torch


def get_device(gpu):
    device = torch.device("cpu")
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    return device


def save_npy_record(npy_path, record, name=None):
    if name == None:
        name = "record"
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    server_name = socket.gethostname()
    np.save(npy_path + "/{}_{}_{}.npy".format(name, server_name, timestamp), record)


def write_avg_json(log_dir, name="record", tags=[]):
    pt_path = os.path.join(log_dir, "results")
    data_list = []
    files = []
    for filename in os.listdir(pt_path):
        if filename.startswith(name) and filename.endswith(".npy"):
            file_path = os.path.join(pt_path, filename)
            data = np.load(file_path, allow_pickle=True).item()
            data_list.append(data)
            files.append(filename)

    all_records = {}
    for tag in tags:
        all_records[tag] = [float(d[tag]) for d in data_list]

    results = {}
    results["avg"] = {k: float(np.array(v).mean()) for k, v in all_records.items()}
    results["std"] = {k: float(np.array(v).std()) for k, v in all_records.items()}
    results["all"] = all_records
    results["all"]["files"] = files
    json_path = os.path.join(log_dir, name + ".json")
    with open(json_path, "w+") as f:
        json.dump(results, f, indent=4)
