# FedMUD

**The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning [ICML 2025]**

## Environment Setup
Please execute the following command to install the necessary dependencies:

We recommend using `uv` to build your Python environment. If you're unfamiliar with `uv`， you can find an introduction and installation guide [here](https://docs.astral.sh/uv/).

```bash
uv sync
source .venv/bin/activate
```

Alternatively， you can also use traditional `pip` instructions directly to install the necessary dependencies.

```bash
pip install -r requirements.txt
```

## Data Partition
Run the following code to download and partition datasets:
```bash
cd dataloader/
python ./dataloader/datapartition.py
```

## Run All Experiments
`./run/` contains scripts to run the experiments, for example:

```bash
./run/12-feddmu.sh
```

## Citation

```
@inproceedings{li2025fedmud,
  author       = {Shiwei Li and
                  Xiandi Luo and
                  Haozhao Wang and
                  Xing Tang and
                  Shijie Xu and
                  Weihong Luo and
                  Yuhua Li and
                  Xiuqiang He and
                  Ruixuan Li},
  title        = {The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning.},
  booktitle    = {The Forty-Second International Conference on Machine Learning, {ICML 2025}, Vancouver, Canada, 13th-19th July, 2025},
  year         = {2025},
}
```
