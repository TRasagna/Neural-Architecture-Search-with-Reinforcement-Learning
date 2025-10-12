#!/usr/bin/env python3
"""Download datasets."""

from pathlib import Path
from torchvision import datasets

def download_cifar10():
    data_dir = Path("data/cifar10")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading CIFAR-10...")
    datasets.CIFAR10(root=str(data_dir), train=True, download=True)
    datasets.CIFAR10(root=str(data_dir), train=False, download=True)
    print("CIFAR-10 downloaded!")

if __name__ == "__main__":
    download_cifar10()
