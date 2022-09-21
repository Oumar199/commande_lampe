"""Entraînement d'un modèle plus simple.
"""
from speech_google.modelization.analyze.tensorboard import SpecTensorboard
from speech_google.modelization.models.simple_spec_models import SpecModel2
from speech_google.modelization.training.run_training import SpecRunner1
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

class SpecRunner3(SpecRunner1):
    def __init__(self, dataset: Union[Dataset, ImageFolder], version: int, running_directory: str = "led_runs", means: List = [0.5], stds: List = [0.5]):
        """Initialisation de quelques paramètres

        Args:
            dataset (Union[Dataset, ImageFolder]): L'ensemble de données 
            version (int): La version du modèle ou de la classe 
            running_directory (str, optional): Le chemin vers le dossier de sauvegarde des données de tensorboard. Defaults to "led_runs".
            means (List): Une liste de moyennes. Defaults to [0.5].
            stds (List): Une liste d'écart-types.Defaults to [0.5].
        """
        super().__init__(dataset, version, running_directory, means, stds)
    
    def split_dataset(self, test_size: float = 0.2, seed: Union[int, None] = None):
        """Cette méthode nous fournit un dataset d'entraînement et un autre de test

        Args:
            test_size (float, optional): La proportion des données de test. Defaults to 0.2.
            seed (Union[int, None], optional): La graine du générateur de nombre. Defaults to None.
        """
        if seed: torch.manual_seed(seed) 
        
        n_test = round(len(self.dataset)*0.2)
        n_train = len(self.dataset) - n_test
        return random_split(self.dataset, [n_train, n_test])
    
    def compile(self, drop_out_rate: float = 0.3, num_units: int = 3000, output_size: int = 7, learning_rate: float = 0.0001, batch_size: int = 5):
        """Initialisation des paramètres d'entraînement

        Args:
            drop_out_rate (float): Le taux de drop out. Defaults to 0.3.
            num_units (int, optional): Nombre d'unités de la première couche dense. Defaults to 300.
            output_size (int, optional): Nombre de classes. Defaults to 2.
            learning_rate (float, optional): Le taux d'apprentissage. Defaults to 0.0001.
            batch_size (int, optional): La taille d'un lot d'images. Defaults to 5.
        """
        
        # Récupérons d'abord le type de device ('gpu cuda' ou 'cpu') présent sur l'appareil
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Attribution des hyper paramètres et du batch size
        self.drop_out_rate = drop_out_rate
        self.num_units = num_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Initialisation du modèle et de ses poids
        self.model: nn.Module = SpecModel2(drop_out_rate, num_units, output_size)
        self.model.to(self.device)
        
        # Initialisation de l'optimiseur
        self.optimizer = Adam(self.model.parameters(), lr = learning_rate)
        
        if self.dataset:
            # Récupérons les données d'entraînement et de test
            self.train_set, self.test_set = self.split_dataset()
            
            # Récupérons les générateurs de données
            self.train_loader = DataLoader(self.train_set, batch_size = batch_size, shuffle = True)
            self.test_loader = DataLoader(self.test_set, batch_size = batch_size, shuffle=False)
            
            # Initialisation des closeurs
            self.train_steps = self.batch_train(self.model, self.optimizer)
            self.test_steps = self.batch_test(self.model)
        
    def get_hparams_dict(self):
        """Récupération des hyper paramètres du modèle
        """
        return {
            "drop_out_rate": self.drop_out_rate,
            "num_units": self.num_units,
            "learning_rate": self.learning_rate
        }
    
    def train(self, epochs: int = 50):
        """Entraînement du modèle

        Args:
            epochs (int, optional): Le nombre d'itérations. Defaults to 50.
        """
        
        # Initialisation des métriques
        self.train_accuracy = self.test_accuracy = 0
        train_accuracies = test_accuracies = train_losses = test_losses = 0
        n_train = n_test = 0 
        
        for epoch in tqdm(range(epochs)):
            for mode in ['train', 'test']:
                acc = 0
                if mode == 'train':
                    loader = self.train_loader
                    batch_step = self.train_steps
                    self.model.train() 
                    n_batches = n_train
                    losses = train_losses
                    accuracies = train_accuracies
                else:
                    self.model.eval()
                    loader = self.test_loader
                    batch_step = self.test_steps
                    n_batches = n_test
                    losses = test_losses
                    accuracies = test_accuracies
                    
                with torch.set_grad_enabled(mode == 'train'):    
                    for inputs, labels in loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device) 
                        
                        # Récupération de la perte et des sorties
                        loss, outputs = batch_step(inputs, labels)
                        
                        # Récupérons les prédictions
                        _, predictions = torch.max(outputs.data, dim=1)
                        
                        # Ajout du loss
                        losses += loss
                        
                        # Ajoutons l'accuracy
                        accuracies += (predictions == labels).sum().item()  
                        acc += (predictions == labels).sum().item()
                        
                        if n_batches % 15 == 0:
                            self.writer.add_scalar(f'{mode}_accuracy', accuracies*100 / (15*self.batch_size), global_step=n_batches)    
                            self.writer.add_scalar(f'{mode}_losses', losses/15, global_step=n_batches)    
                            losses = 0
                            accuracies = 0
                            
                        n_batches += 1
                
                if mode == 'train':
                    n_train = n_batches
                    train_losses = losses
                    train_accuracies = accuracies
                    self.train_accuracy = acc * 100 / (self.batch_size * len(loader))
                else:
                    n_test = n_batches
                    test_losses = losses
                    test_accuracies = accuracies
                    self.test_accuracy = acc * 100 / (self.batch_size * len(loader))
                
    def save_model(self, saving_path: str = "speech_google/modelization/storages"):
        """Sauvegarde du modèle et de quelques paramètres

        Args:
            saving_path (str, optional): Le chemin vers le dossier de sauvegarde. Defaults to "speech_google/storages".
        """
        path = os.path.join(saving_path, f"version{self.version}")
        if not os.path.exists(path):
            os.makedirs(path)
        
        checkpoints = {
            "model_state_dict": self.model.state_dict(),
            "hparams_dict": self.get_hparams_dict()
        }
        
        torch.save(checkpoints, os.path.join(path, "checkpoints.pth"))
    
    def predict(self, data: Union[torch.Tensor, np.ndarray], classes: List[Union[str, int]]):
        """Effectuer une prédiction sur des données fournies

        Args:
            data (Union[torch.Tensor, np.ndarray]): Les données sur les-quelles sont effectuées les prédictions
            classes (List[Union[str, int]]): Les différentes classes possibles.
        """
        
        # Transformons les données si elles sont de type numpy array
        if type(data) is np.ndarray:
            data = torch.from_numpy(data)
        
        # Ajoutons une dimension aux données si le nombre de dimensions est inférieur à 4
        if len(data.size()) < 4:
            data = data.unsqueeze(0)
        
        outputs = self.model(data)
        
        predictions = torch.round(outputs)
        
        print(predictions)
        # return [classes[prediction] for prediction in predictions.tolist()]
