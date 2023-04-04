"""Entraînement d'un modèle plus complexe.
"""
from speech_google.modelization.analyze.tensorboard2 import SpecTensorboard2
from speech_google.modelization.models.mel_spec_models import SpecModel3
from speech_google.modelization.training.simple_run_training import SpecRunner3
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torch.optim import Adam, SGD
from typing import Union, List
from tqdm import tqdm
from torch import nn
import numpy as np
import torch
import os

class SpecRunner4(SpecTensorboard2, SpecRunner3):
    def __init__(self, dataset: Union[Dataset, ImageFolder], version: int, running_directory: str = "led_runs", means: List = [0.5], stds: List = [0.5]):
        """Initialisation de quelques paramètres

        Args:
            dataset (Union[Dataset, ImageFolder]): L'ensemble de données 
            version (int): La version du modèle ou de la classe 
            running_directory (str, optional): Le chemin vers le dossier de sauvegarde des données de tensorboard. Defaults to "led_runs".
            means (List): Une liste de moyennes. Defaults to [0.5].
            stds (List): Une liste d'écart-types.Defaults to [0.5].
        """
        SpecRunner3.__init__(self, dataset, version, running_directory, means, stds)
        SpecTensorboard2.__init__(self, dataset, version, running_directory)
    
    def compile(self, input_shape: tuple = (308, 465), o_channel: tuple = (16, 32), drop_out_rate: float = 0.3, num_units: int = 3000, output_size: int = 7, learning_rate: float = 0.0001, batch_size: int = 5, train_accuracy_thres: Union[float, None] = None, val_accuracy_thres: Union[float, None] = None, train_auroc_thres: Union[float, None] = None, val_auroc_thres: Union[float, None] = None, weight_decay: float = 0, seed: int = 50):
        """Initialisation des paramètres d'entraînement

        Args:
            input_shape (tuple): Un tuple contenant les dimensions (hauteur et largeur) des données d'entrées. Defaults to (308, 465).
            o_channel (tuple): Un tuple indiquant le nombre de chaines de sortie pour chaque couche de convolution. Defaults to (16, 32).
            drop_out_rate (float): Le taux de drop out. Defaults to 0.3.
            num_units (int, optional): Nombre d'unités de la première couche dense. Defaults to 300.
            output_size (int, optional): Nombre de classes. Defaults to 2.
            learning_rate (float, optional): Le taux d'apprentissage. Defaults to 0.0001.
            batch_size (int, optional): La taille d'un lot d'images. Defaults to 5.
            train_accuracy_thres (float): Le seuil minimal de précision pour les données d'entraînement. Defaults to None.
            test_accuracy_thres (float): Le seuil minimal de précision pour les données de test. Defaults to None.
            train_auroc_thres (float, optional): Le seuil minimal de l'auroc pour les données d'entraînement. Defaults to None.
            val_auroc_thres (float, optional): Le seuil minimal de l'auroc pour les données de test. Defaults to None.
            weight_decay (float, optional): La dégradation des pondérations. Defaults to 0.
            seed (int): Graine du generateur. Defaults to 50.
        """
        generator = torch.manual_seed(seed)
        # Récupérons d'abord le type de device ('gpu cuda' ou 'cpu') présent sur l'appareil
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialisation de l'instance de la métrique auroc
        # self.multiclass_auroc = MulticlassAUROC(num_classes=output_size, average="macro", thresholds=None)
        
        # Attribution des hyper paramètres et du batch size
        self.drop_out_rate = drop_out_rate
        self.num_units = num_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_accuracy_thres = train_accuracy_thres
        self.val_accuracy_thres = val_accuracy_thres
        self.train_auroc_thres = train_auroc_thres
        self.val_auroc_thres = val_auroc_thres
        
        # Initialisation du modèle et de ses poids
        self.model: nn.Module = SpecModel3(input_shape, o_channel, drop_out_rate, num_units, output_size, seed = seed)
        self.model.to(self.device)
        
        # Initialisation de l'optimiseur
        self.optimizer = Adam(self.model.parameters(), lr = learning_rate, weight_decay = weight_decay)
        
        if self.dataset:
            # Récupérons les données d'entraînement et de test
            self.train_set, self.test_set = self.split_dataset(seed = seed)
            
            # Récupérons les générateurs de données
            self.train_loader = DataLoader(self.train_set, batch_size = batch_size, shuffle = True, generator = generator)
            self.test_loader = DataLoader(self.test_set, batch_size = batch_size, shuffle = False, generator = generator)
            
            # Initialisation des closeurs
            self.train_steps = self.batch_train(self.model, self.optimizer)
            self.test_steps = self.batch_test(self.model)
        
    
