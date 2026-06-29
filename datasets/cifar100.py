# datasets/cifar100.py

from torchvision.datasets import CIFAR100

from .transforms import (
    vit_train_transform,
    vit_test_transform,
)

from .splitter import split_by_classes


def get_tasks(cfg):

    train = CIFAR100(
        root=cfg["data_root"],
        train=True,
        download=True,
        transform=vit_train_transform(),
    )

    test = CIFAR100(
        root=cfg["data_root"],
        train=False,
        download=True,
        transform=vit_test_transform(),
    )

    return split_by_classes(
        train,
        test,
        cfg["num_tasks"],
        cfg["classes_per_task"],
    )
