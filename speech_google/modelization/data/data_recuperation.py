"""Module de récupération de données avec transformations.
"""
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from typing import Union, List
from torchvision.transforms import Compose
import os


def get_spectrograms(path: str, transformers: Compose):
    """Récupération des spectrogrammes avec ImageFolder de pytorch

    Args:
        path (str): Chemin vers les spectrogrammes
        transformers (Compose): Les transformations a effectuées sur les spectrogrammes
    """
    if os.path.exists(path):
        dataset = ImageFolder(path, transform=transformers)
        return dataset
    else:
        raise OSError("Le chemin spécifié est introuvale !")
