import torch
import torch.nn as nn
from math import ceil, floor
import torch.nn.functional as F
from torch.nn.modules.container import Sequential


def kron_3d(A, B):
    result = torch.einsum("oim,ojn->oijmn", A, B)
    result = result.view(B.shape[0], B.shape[1] ** 2, B.shape[2] ** 2)
    return result


class Mat_Weight(nn.Module):
    def __init__(
        self, in_features, out_features, rank, init_type_mag, requires_grad="left-right"
    ):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.rank = rank
        self.init_type_mag = init_type_mag
        self.requires_grad = requires_grad
        self.left = nn.Parameter(
            torch.zeros(size=(out_features, self.rank)),
            requires_grad=bool("left" in requires_grad),
        )
        self.right = nn.Parameter(
            torch.zeros(size=(self.rank, in_features)),
            requires_grad=bool("right" in requires_grad),
        )
        self.init_parameters()

    def init_parameters(self, seed=1234):
        generator = torch.Generator(device=self.left.device).manual_seed(seed)
        mark = "_" if "_" in self.init_type_mag else "-"
        mag = float(self.init_type_mag.split(mark)[1])

        if "nor" in self.init_type_mag:
            if "right" in self.requires_grad:
                nn.init.normal_(self.left, 0, mag, generator=generator)
                nn.init.zeros_(self.right)
            else:
                nn.init.normal_(self.right, 0, mag, generator=generator)
                nn.init.zeros_(self.left)
        elif "uni" in self.init_type_mag:
            if "right" in self.requires_grad:
                nn.init.uniform_(self.left, -mag, mag, generator=generator)
                nn.init.zeros_(self.right)
            else:
                nn.init.uniform_(self.right, -mag, mag, generator=generator)
                nn.init.zeros_(self.left)
        else:
            raise ValueError(f"wrong init_type_mag: {self.init_type_mag}.")

    def forward(self):
        return torch.mm(self.left, self.right)


class Mat_Update(nn.Module):
    def __init__(self, in_shape, out_shape, config=None):
        super().__init__()
        self.in_shape, self.out_shape = in_shape, out_shape
        self.dmu_pattern = config["dmu_pattern"]
        self.init_type_mag = config["init_type_mag"]
        self.rank = self.get_rank(config["ratio"])
        self.generate_weight()

    def get_rank(self, ratio):
        num_weights = self.out_shape * self.in_shape
        if self.dmu_pattern in ["FAB", "fab"]:
            inout = self.in_shape
        else:
            inout = self.out_shape + self.in_shape
        rank = ceil(num_weights * ratio / inout)
        return max(rank, 1)

    def generate_weight(self):
        if self.dmu_pattern in ["AB", "ab"]:
            self.W = Mat_Weight(
                self.in_shape,
                self.out_shape,
                self.rank,
                self.init_type_mag,
                "left-right",
            )
        elif self.dmu_pattern in ["FAB", "fab"]:
            self.W = Mat_Weight(
                self.in_shape, self.out_shape, self.rank, self.init_type_mag, "right"
            )
        elif self.dmu_pattern in ["FAB+CFD", "fab+cfd"]:
            self.W1 = Mat_Weight(
                self.in_shape, self.out_shape, self.rank, self.init_type_mag, "right"
            )
            self.W2 = Mat_Weight(
                self.in_shape, self.out_shape, self.rank, self.init_type_mag, "left"
            )
        else:
            raise ValueError("wrong dmu_pattern.")

    def init_parameters(self, seed=1234):
        if self.dmu_pattern in ["FAB+CFD", "fab+cfd"]:
            self.W1.init_parameters(seed)
            self.W2.init_parameters(seed + 42)
        else:
            self.W.init_parameters(seed)

    def forward(self):
        if self.dmu_pattern in ["FAB+CFD", "fab+cfd"]:
            weight = self.W1() + self.W2()
        else:
            weight = self.W()
        return weight


class Kron_Weight(nn.Module):
    def __init__(self, rank, size, init_type_mag, requires_grad="left-right"):
        super().__init__()
        self.rank = rank
        self.size = size
        self.init_type_mag = init_type_mag
        self.requires_grad = requires_grad
        self.left = nn.Parameter(
            torch.zeros(size=(rank, size, size)),
            requires_grad=bool("left" in requires_grad),
        )
        self.right = nn.Parameter(
            torch.zeros(size=(rank, size, size)),
            requires_grad=bool("right" in requires_grad),
        )
        self.init_parameters()

    def init_parameters(self, seed=1234):
        generator = torch.Generator(device=self.left.device).manual_seed(seed)
        mark = "_" if "_" in self.init_type_mag else "-"
        mag = float(self.init_type_mag.split(mark)[1])

        if "nor" in self.init_type_mag:
            if "right" in self.requires_grad:
                nn.init.normal_(self.left, 0, mag, generator=generator)
                nn.init.zeros_(self.right)
            else:
                nn.init.normal_(self.right, 0, mag, generator=generator)
                nn.init.zeros_(self.left)
        elif "uni" in self.init_type_mag:
            if "right" in self.requires_grad:
                nn.init.uniform_(self.left, -mag, mag, generator=generator)
                nn.init.zeros_(self.right)
            else:
                nn.init.uniform_(self.right, -mag, mag, generator=generator)
                nn.init.zeros_(self.left)
        else:
            raise ValueError(f"wrong init_type_mag: {self.init_type_mag}.")

    def forward(self):
        return kron_3d(self.left, self.right)


class Kron_Update(nn.Module):
    def __init__(self, in_shape, out_shape, config=None):
        super().__init__()
        self.in_shape, self.out_shape = in_shape, out_shape
        self.dmu_pattern = config["dmu_pattern"]
        self.init_type_mag = config["init_type_mag"]
        self.rank, self.size = self.get_rank_size(config["ratio"])
        self.generate_weight()

    def get_rank_size(self, ratio):
        num_weights = self.out_shape * self.in_shape
        if self.dmu_pattern in ["FAB", "fab"]:
            ratio = ratio * 2

        rank = ceil((ratio**2) * num_weights / 4)
        size = ceil((num_weights / rank) ** 0.25)
        return rank, size

    def generate_weight(self):
        if self.dmu_pattern in ["AB", "ab"]:
            self.W = Kron_Weight(self.rank, self.size, self.init_type_mag, "left-right")
        elif self.dmu_pattern in ["FAB", "fab"]:
            self.W = Kron_Weight(self.rank, self.size, self.init_type_mag, "right")
        elif self.dmu_pattern in ["FAB+CFD", "fab+cfd"]:
            self.W1 = Kron_Weight(self.rank, self.size, self.init_type_mag, "right")
            self.W2 = Kron_Weight(self.rank, self.size, self.init_type_mag, "left")
        else:
            raise ValueError("wrong dmu_pattern.")

    def init_parameters(self, seed=1234):
        if self.dmu_pattern in ["FAB+CFD", "fab+cfd"]:
            self.W1.init_parameters(seed)
            self.W2.init_parameters(seed + 42)
        else:
            self.W.init_parameters(seed)

    def forward(self):
        if self.dmu_pattern in ["FAB+CFD", "fab+cfd"]:
            weight = self.W1() + self.W2()
        else:
            weight = self.W()
        weight = weight.reshape(-1)[: self.in_shape * self.out_shape]
        return weight.reshape(self.out_shape, self.in_shape)


class DMU_Linear(nn.Module):
    def __init__(self, in_features, out_features, config=None):
        super(DMU_Linear, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(
            torch.zeros((self.out_features, self.in_features)), requires_grad=False
        )
        nn.init.normal_(self.weight, mean=0, std=0.01)

        self.dmu_type = config["dmu_type"]
        if self.dmu_type == "mat":
            self.update = Mat_Update(in_features, out_features, config)
        elif self.dmu_type == "kron":
            self.update = Kron_Update(in_features, out_features, config)
        else:
            raise ValueError("Unknown dmu_type")

    def push_reset_update(self, push=True, seed=1234):
        if push:
            with torch.no_grad():
                self.weight.data += self.update()
            self.update.init_parameters(seed)

    def forward(self, x):
        update = self.update()
        return F.linear(x, self.weight + update, None)


class DMU_Conv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, config=None
    ):
        super(DMU_Conv2d, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.weight = nn.Parameter(
            torch.zeros((out_channels, in_channels, kernel_size, kernel_size)),
            requires_grad=False,
        )
        nn.init.normal_(self.weight, mean=0, std=0.01)

        self.dmu_type = config["dmu_type"]
        if self.dmu_type == "mat":
            self.update = Mat_Update(
                in_channels * kernel_size, out_channels * kernel_size, config
            )
        elif self.dmu_type == "kron":
            self.update = Kron_Update(
                in_channels * kernel_size, out_channels * kernel_size, config
            )
        else:
            raise ValueError("Unknown dmu_type")

    def push_reset_update(self, push=True, seed=1234):
        if push:
            with torch.no_grad():
                self.weight.data += self.update().reshape(
                    self.out_channels,
                    self.in_channels,
                    self.kernel_size,
                    self.kernel_size,
                )
            self.update.init_parameters(seed)

    def forward(self, x):
        update = self.update().reshape(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        return F.conv2d(
            x, self.weight + update, stride=self.stride, padding=self.padding
        )


def dmu_replace_modules(model, config=None, skip_layers=[], prefix=""):
    for n, m in model._modules.items():
        full_name = f"{prefix}.{n}" if prefix else n
        if full_name not in skip_layers:
            if isinstance(m, nn.Conv2d):
                setattr(
                    model,
                    n,
                    DMU_Conv2d(
                        m.in_channels,
                        m.out_channels,
                        m.kernel_size[0],
                        m.stride[0],
                        m.padding[0],
                        config,
                    ),
                )
            if isinstance(m, nn.Linear):
                setattr(model, n, DMU_Linear(m.in_features, m.out_features, config))
            if isinstance(m, (Sequential)):
                dmu_replace_modules(m, config, skip_layers, full_name)


def dmu_push_reset_update(model, config, seeds=None, prefix_index=0):
    if seeds is None:
        generator = torch.Generator().manual_seed(config["seed"])
        seeds = torch.randint(1, 999, (100,), generator=generator)

    for n, m in model._modules.items():
        if isinstance(m, (DMU_Conv2d, DMU_Linear)):
            used_seed = seeds[prefix_index]
            prefix_index += 1
            m.push_reset_update(config["push"], int(used_seed))
        if isinstance(m, (Sequential)):
            prefix_index = dmu_push_reset_update(m, config, seeds, prefix_index)
    return prefix_index
