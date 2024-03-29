"""Classe d'apprentissage de réseau de neurones
"""
from speech_google.modelization.analyze.tensorboard import SpecTensorboard

from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter # type: ignore
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import Adam, SGD
from typing import Union, List
from tqdm import tqdm
from torch import nn
import numpy as np
import torch
import os


class SpecRunner1(SpecTensorboard):
    """Classe d'entraînement du modèle"""

    def __init__(
        self,
        dataset: Union[Dataset, ImageFolder],
        version: int,
        running_directory: str = "runs",
        means: List = [0.5],
        stds: List = [0.5],
    ):
        """Initialisation de quelques paramètres

        Args:
            dataset (Union[Dataset, ImageFolder]): L'ensemble de données
            version (int): La version du modèle ou de la classe
            running_directory (str, optional): Le chemin vers le dossier de sauvegarde des données de tensorboard. Defaults to "runs".
            means (List): Une liste de moyennes. Defaults to [0.5].
            stds (List): Une liste d'écart-types.Defaults to [0.5].
        """
        super().__init__(dataset, version, running_directory)
        self.means = means
        self.stds = stds

    def batch_train(
        self,
        model: nn.Module,
        optimizer: Union[Adam, SGD],
        loss_function: Union[
            CrossEntropyLoss, BCEWithLogitsLoss, BCELoss
        ] = CrossEntropyLoss(),
    ):
        """Un closeur permettant d'effectuer un entraînement sur un batch

        Args:
            model (nn.Module): Le modèle
            optimizer (Union[Adam, SGD]): L'optimiseur
            loss_function (Union[CrossEntropyLoss, BCEWithLogitsLoss, BCELoss]): La fonction de perte. Defaults to CrossEntropyLoss().
        """

        def wrapper(inputs: torch.Tensor, labels: torch.Tensor):
            # Récupérons la sortie du modèle après qu'on l'ai fourni les entrées
            outputs = model(inputs)

            # Calculons la perte
            loss = loss_function(outputs, labels)

            # On détermine les gradients
            loss.backward()

            # On optimise les paramètres
            optimizer.step()

            # Reinitilisation des gradients
            optimizer.zero_grad()

            return loss, outputs

        return wrapper

    def batch_test(
        self,
        model: nn.Module,
        loss_function: Union[
            CrossEntropyLoss, BCEWithLogitsLoss, BCELoss
        ] = CrossEntropyLoss(),
    ):
        """Un closeur permettant d'effectuer un test sur un batch

        Args:
            model (nn.Module): Le modèle
            loss_function (Union[CrossEntropyLoss, BCEWithLogitsLoss, BCELoss]): La fonction de perte. Defaults to CrossEntropyLoss().
        """

        def wrapper(inputs: torch.Tensor, labels: torch.Tensor):
            # Récupérons la sortie du modèle après qu'on l'ai fourni les entrées
            outputs = model(inputs)

            # Calculons la perte
            loss = loss_function(outputs, labels)

            return loss, outputs

        return wrapper

    def compile(
        self,
        drop_out_rate: float = 0.3,
        num_units1: int = 3000,
        num_units2: int = 1000,
        num_units3: int = 300,
        output_size: int = 7,
        learning_rate: float = 0.0001,
        batch_size: int = 5,
    ):
        """Initialisation des paramètres d'entraînement

        Args:
            drop_out_rate (float): Le taux du drop out. Defaults to 0.3.
            num_units1 (int, optional): Nombre d'unités de la première couche dense. Defaults to 3000.
            num_units2 (int, optional): Nombre d'unités de la deuxième couche dense. Defaults to 1000.
            num_units2 (int, optional): Nombre d'unités de la troisième couche dense. Defaults to 300.
            output_size (int, optional): Nombre d'unités de la quatrième couche dense. 7.
            learning_rate (float, optional): Le taux d'apprentissage. Defaults to 0.0001.
            batch_size (int, optional): La taille d'un lot d'images. Defaults to 5.
        """
        pass

    def get_hparams_dict(self):
        """Récupération des hyper paramètres du modèle"""
        pass

    def train(self, epochs: int = 50):
        """Entraînement du modèle

        Args:
            epochs (int, optional): Le nombre d'itérations. Defaults to 50.
        """
        pass

    def save_model(self, saving_path: str = "speech_google/modelization/storages"):
        """Sauvegarde du modèle et de quelques paramètres

        Args:
            saving_path (str, optional): Le chemin vers le dossier de sauvegarde. Defaults to "speech_google/storages".
        """
        pass

    def load_model(self, saving_path: str = "speech_google/modelization/storages"):
        """Chargement du modèle

        Args:
            saving_path (str, optional): Le chemin vers le dossier ou le modèle est sauvegardé. Defaults to "speech_google/modelization/storages".
        """
        path = os.path.join(saving_path, f"version{self.version}/checkpoints.pth")
        if os.path.exists(path):
            self.checkpoints = torch.load(path, map_location=torch.device("cpu"))
            self.model.load_state_dict(self.checkpoints["model_state_dict"])

    def predict(
        self, data: Union[torch.Tensor, np.ndarray], classes: List[Union[str, int]]
    ):
        """Effectuer une prédiction sur des données fournies

        Args:
            data (Union[torch.Tensor, np.ndarray]): Les données sur les-quelles sont effectuées les prédictions
            classes (List[Union[str, int]]): Les différentes classes possibles.
        """

        # Transformons les données si elles sont de type numpy array
        if type(data) is np.ndarray:
            data = torch.from_numpy(data)

        # Ajoutons une dimension aux données si le nombre de dimensions est inférieur à 4
        if len(data.size()) < 4: # type: ignore
            data = data.unsqueeze(0) # type: ignore

        outputs = self.model(data)

        _, predictions = torch.max(outputs.data, 1)

        return [classes[prediction] for prediction in predictions.tolist()]


# ----------------------------------------------
from torch.nn import BCELoss


class SpecRunner2(SpecRunner1):
    def __init__(
        self,
        dataset: Union[Dataset, ImageFolder],
        version: int,
        running_directory: str = "led_runs",
        means: List = [0.5],
        stds: List = [0.5],
    ):
        """Initialisation de quelques paramètres

        Args:
            dataset (Union[Dataset, ImageFolder]): L'ensemble de données
            version (int): La version du modèle ou de la classe
            running_directory (str, optional): Le chemin vers le dossier de sauvegarde des données de tensorboard. Defaults to "runs".
            means (List): Une liste de moyennes. Defaults to [0.5].
            stds (List): Une liste d'écart-types.Defaults to [0.5].
        """
        super().__init__(dataset, version, running_directory, means, stds)

    def compile(
        self,
        drop_out_rate: float = 0.3,
        num_units1: int = 3000,
        num_units2: int = 1000,
        num_units3: int = 300,
        output_size: int = 7,
        learning_rate: float = 0.0001,
        batch_size: int = 5,
    ):
        """Initialisation des paramètres d'entraînement

        Args:
            drop_out_rate (float): Le taux du drop out. Defaults to 0.3.
            num_units1 (int, optional): Nombre d'unités de la première couche dense. Defaults to 3000.
            num_units2 (int, optional): Nombre d'unités de la deuxième couche dense. Defaults to 1000.
            num_units2 (int, optional): Nombre d'unités de la troisième couche dense. Defaults to 300.
            output_size (int, optional): Nombre d'unités de la quatrième couche dense. 7.
            learning_rate (float, optional): Le taux d'apprentissage. Defaults to 0.0001.
            batch_size (int, optional): La taille d'un lot d'images. Defaults to 5.
        """
        pass

    def train(self, epochs: int = 50):
        """Entraînement du modèle

        Args:
            epochs (int, optional): Le nombre d'itérations. Defaults to 50.
        """
        pass

    def predict(
        self, data: Union[torch.Tensor, np.ndarray], classes: List[Union[str, int]]
    ):
        """Effectuer une prédiction sur des données fournies

        Args:
            data (Union[torch.Tensor, np.ndarray]): Les données sur les-quelles sont effectuées les prédictions
            classes (List[Union[str, int]]): Les différentes classes possibles.
        """

        pass
