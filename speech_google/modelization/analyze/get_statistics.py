"""Ce module va nous permettre de déterminer les statistiques d'un ensemble (pytorch Dataset) d'images
"""
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from typing import Union


def get_mean_std(images: Union[ImageFolder, Dataset], on_each_color: bool = False):
    """Détermine les moyennes et les écart-types d'un ensemble d'images RVB fourni.

    Args:
        images (Union[ImageFolder, Dataset]): L'ensemble d'images
    """
    # chargement de données
    loader = DataLoader(images, batch_size=len(images)) # type: ignore

    data, _ = iter(loader).next()

    # calculons les moyennes et les écart-types
    means = data.mean([0, 2, 3]).tolist() if on_each_color else data.mean().item()
    stds = data.std([0, 2, 3]).tolist() if on_each_color else data.std().item()
    print(
        f"""
          Data Size : {data.size()}
          Data mean : {means}
          Data STD : {stds}
          """
    )
    return means, stds
