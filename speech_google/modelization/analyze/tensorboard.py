"""Tensorboard nous permet de visualiser les données, les métriques, l'architecture ainsi que beaucoup d'autres choses afin de 
juger de la performance du modèle et d'effectuer des analyses plus poussées. Ce module contient des classes et des fonctions
capable de charger des données au niveau de tensorboard.
"""
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from typing import Union
import os

class SpecTensorboard(object):
    """Une classe tensorboard uniquement utilisée pour le modèle d'entraînement des spectogramems
    """
    def __init__(self, dataset: Union[Dataset, ImageFolder], version: int, running_directory: str = "runs"):
        """Initialisation du tensorboard

        Args:
            version (int): Version du tensorboard voir du modèle.   
            running_directory (str, optional): Le chemin vers le dossier de sauvegarde des données de tensorboard. Defaults to "runs".
        """
        self.running_directory = running_directory
        self.path = os.path.join(self.running_directory, f"version{version}")
        self.version = version
        
        self.writer = SummaryWriter(self.path)
        self.model = None
        self.dataset = dataset
        self.train_accuracy = None
        self.test_accuracy = None
    
    def add_images(self, n_images: int = 28, seed: Union[int, None] = None):
        """Chargement d'images au niveau de tensorboard

        Args:
            n_images (int): Le nombre d'images à charger.Defaults to 28.
            seed (Union[int, None]): Graine du générateur.
        """
        torch.manual_seed(torch.randint(0, 1000, (1, 1)) if not seed else seed)
        
        # initialisation d'un chargeur de données
        loader = DataLoader(self.dataset, batch_size = n_images)
        
        # chargement d'un batch d'images
        data, _ = iter(loader).next()
        
        # chargement des données au niveau de tensorboard
        self.writer.add_images("spectrogrammes", data)
    
    def add_architecture(self):
        """Charger l'architecture du modèle dans tensorboard
        """
        loader = DataLoader(dataset=self.dataset, batch_size=40)
        
        data, _ = iter(loader).next()
        
        with SummaryWriter(os.path.join(self.running_directory, 'graph')) as writer:
            writer.add_graph(self.model, data)
            
    def add_grid_results1(self, hparams: dict, output_size: int = 2, batch_size: int = 5, n_epochs: int = 5, grid_name:str = "h_param_results"):
        """Cette fonction permet d'effectuer la recherche par quadrillage sur le modèle simple.

        Args:
            hparams (dict): Un dictionnaire contenant les hyper-paramètres sur les lequels sont effectués les recherches
            output_size (int, optional): Nombre de classes. Defaults to 2.
            batch_size (int, optional): La taille d'un lot d'images. Defaults to 5.
            n_epochs (int): Le nombre d'itérations
            grid_name (str, optional): Le nom du dossier dans lequel seront sauvés les résultats. Defaults to "h_param_results".
        """
        with SummaryWriter(os.path.join(self.path, grid_name)) as w:
            for lr in hparams['learning_rate']:
                for drop_out in hparams['dropout']:
                    for num_units in hparams['num_units']:
                        params = {'learning_rate': lr, 'dropout': drop_out, 'num_units': num_units}
                        self.compile(learning_rate = lr, drop_out_rate = drop_out, num_units = num_units, batch_size=batch_size, output_size=output_size)
                        self.train(n_epochs)
                        w.add_hparams(params, {'train_accuracy': self.train_accuracy, 'test_accuracy': self.test_accuracy})
    
    def train(self, epochs: int = 50):
        """Entraînement du modèle

        Args:
            epochs (int, optional): Le nombre d'itérations. Defaults to 50.
        """
        raise NotImplementedError


    def compile(self, drop_out_rate: float = 0.3, num_units: int = 3000, output_size: int = 7, learning_rate: float = 0.0001, batch_size: int = 5):
        """Initialisation des paramètres d'entraînement

        Args:
            drop_out_rate (float): Le taux de drop out. Defaults to 0.3.
            num_units (int, optional): Nombre d'unités de la première couche dense. Defaults to 300.
            output_size (int, optional): Nombre de classes. Defaults to 2.
            learning_rate (float, optional): Le taux d'apprentissage. Defaults to 0.0001.
            batch_size (int, optional): La taille d'un lot d'images. Defaults to 5.
        """
        raise NotImplementedError
