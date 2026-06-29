# datasets/builder.py

from .cifar100 import get_tasks as cifar100_tasks


def build_dataset(cfg):

    dataset = cfg["dataset"].lower()

    if dataset == "cifar100":
        return cifar100_tasks(cfg)

    raise ValueError(
        f"Unknown dataset {dataset}"
    )
