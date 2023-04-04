"""Tensorboard nous permet de visualiser les données, les métriques, l'architecture ainsi que beaucoup d'autres choses afin de 
juger de la performance du modèle et d'effectuer des analyses plus poussées. Ce module contient des classes et des fonctions
capable de charger des données au niveau de tensorboard.
"""
import torch
from speech_google.modelization.analyze.tensorboard import SpecTensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from typing import Union
import os

class SpecTensorboard2(SpecTensorboard):
    """Une classe tensorboard uniquement utilisée pour le modèle d'entraînement des spectogramems
    """
    def __init__(self, dataset: Union[Dataset, ImageFolder], version: int, running_directory: str = "runs"):
        """Initialisation du tensorboard

        Args:
            version (int): Version du tensorboard voir du modèle.   
            running_directory (str, optional): Le chemin vers le dossier de sauvegarde des données de tensorboard. Defaults to "runs".
        """
        super().__init__(dataset, version, running_directory)
       
    def add_grid_results1(self, hparams: dict, input_shape: tuple = (308, 465), output_size: int = 2, batch_size: int = 5, n_epochs: int = 5, grid_name:str = "h_param_results", seed: Union[int, None] = None):
        """Cette fonction permet d'effectuer la recherche par quadrillage sur le modèle simple.

        Args:
            hparams (dict): Un dictionnaire contenant les hyper-paramètres sur les lequels sont effectués les recherches
            input_shape (tuple): Un tuple contenant les dimensions (hauteur et largeur) des données d'entrées. Defaults to (308, 465).
            output_size (int, optional): Nombre de classes. Defaults to 2.
            batch_size (int, optional): La taille d'un lot d'images. Defaults to 5.
            n_epochs (int): Le nombre d'itérations
            grid_name (str, optional): Le nom du dossier dans lequel seront sauvés les résultats. Defaults to "h_param_results".
            seed (Union[int, None], optional): La graine de l'initialisateur des poids du modèles. Defaults to None.
        """
        if seed: torch.manual_seed(seed)
        with SummaryWriter(os.path.join(self.path, grid_name)) as w:
            for o_channel in hparams['o_channel']:
                for lr in hparams['learning_rate']:
                    for drop_out in hparams['dropout']:
                        for num_units in hparams['num_units']:
                            params = {'learning_rate': lr, 'dropout': drop_out, 'num_units': num_units, 'o_channel': f"{o_channel[0]}-{o_channel[1]}"}
                            self.compile(learning_rate = lr, input_shape = input_shape, o_channel = o_channel,
                                         drop_out_rate = drop_out, num_units = num_units, batch_size=batch_size, output_size=output_size)
                            self.train(n_epochs)
                            w.add_hparams(params, {'train_accuracy': self.train_accuracy, 'test_accuracy': self.test_accuracy, 'train_auroc': self.train_auroc, 'test_auroc': self.test_auroc})
    

    def compile(self, input_shape: tuple = (308, 465), o_channel: tuple = (16, 32), drop_out_rate: float = 0.3, num_units: int = 3000, output_size: int = 7, learning_rate: float = 0.0001, batch_size: int = 5):
        """Initialisation des paramètres d'entraînement

        Args:
            input_shape (tuple): Un tuple contenant les dimensions (hauteur et largeur) des données d'entrées. Defaults to (308, 465).
            o_channel (tuple): Un tuple indiquant le nombre de chaines de sortie pour chaque couche de convolution. Defaults to (16, 32).
            drop_out_rate (float): Le taux de drop out. Defaults to 0.3.
            num_units (int, optional): Nombre d'unités de la première couche dense. Defaults to 300.
            output_size (int, optional): Nombre de classes. Defaults to 2.
            learning_rate (float, optional): Le taux d'apprentissage. Defaults to 0.0001.
            batch_size (int, optional): La taille d'un lot d'images. Defaults to 5.
        """
        raise NotImplementedError
