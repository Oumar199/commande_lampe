"""Classe d'apprentissage de réseau de neurones
"""
from speech_google.modelization.analyze.tensorboard import SpecTensorboard
from speech_google.modelization.models.spec_models import SpecModel1
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
        # Récupérons d'abord le type de device ('gpu cuda' ou 'cpu') présent sur l'appareil
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Attribution des hyper paramètres et du batch size
        self.drop_out_rate = drop_out_rate
        self.num_units1 = num_units1
        self.num_units2 = num_units2
        self.num_units3 = num_units3
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Initialisation du modèle et de ses poids
        self.model: nn.Module = SpecModel1(
            drop_out_rate, num_units1, num_units2, num_units3, output_size
        )
        self.model.to(self.device)

        # Initialisation de l'optimiseur
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        if self.dataset:
            # Récupérons le chargeur de données
            self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

            # Initialisation du closeur d'entraînement
            self.batch_step = self.batch_train(self.model, self.optimizer)

    def get_hparams_dict(self):
        """Récupération des hyper paramètres du modèle"""
        return {
            "drop_out_rate": self.drop_out_rate,
            "num_units1": self.num_units1,
            "num_units2": self.num_units2,
            "num_units3": self.num_units3,
            "learning_rate": self.learning_rate,
        }

    def train(self, epochs: int = 50):
        """Entraînement du modèle

        Args:
            epochs (int, optional): Le nombre d'itérations. Defaults to 50.
        """
        self.epochs = epochs
        self.model.train()
        accuracies = 0
        losses = 0
        n_batches = 0
        for epoch in tqdm(range(self.epochs)):
            for inputs, labels in self.loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Récupération de la perte et des sorties
                loss, outputs = self.batch_step(inputs, labels)

                # Récupérons les prédictions
                _, predictions = torch.max(outputs.data, dim=1)

                # Ajout du loss
                losses += loss

                # Ajoutons l'accuracy
                accuracies += (predictions == labels).sum().item()

                if n_batches % 100 == 0:
                    self.writer.add_scalar(
                        "accuracy",
                        accuracies * 100 / (100 * self.batch_size),
                        global_step=n_batches,
                    )
                    self.writer.add_scalar(
                        "losses", losses / 100, global_step=n_batches
                    )
                    losses = 0
                    accuracies = 0

                n_batches += 1

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
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hparams_dict": self.get_hparams_dict(),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": self.device,
        }

        torch.save(checkpoints, os.path.join(path, "checkpoints.pth"))

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
from speech_google.modelization.models.led_model import modify_multi_to_binary
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
        super().compile(
            drop_out_rate,
            num_units1,
            num_units2,
            num_units3,
            output_size,
            learning_rate,
            batch_size,
        )
        self.model = modify_multi_to_binary(self.model)
        self.optimizer = Adam(self.model.parameters())
        if self.dataset:
            self.batch_step = self.batch_train(self.model, self.optimizer, BCELoss())

    def train(self, epochs: int = 50):
        """Entraînement du modèle

        Args:
            epochs (int, optional): Le nombre d'itérations. Defaults to 50.
        """
        self.epochs = epochs
        self.model.train()
        accuracies = 0
        losses = 0
        n_batches = 0
        for epoch in tqdm(range(self.epochs)):
            for inputs, labels in self.loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()

                # Récupération de la perte et des sorties
                loss, outputs = self.batch_step(inputs, labels.unsqueeze(1))

                # Récupérons les prédictions
                predictions = torch.round(outputs).squeeze(1)

                # Ajout du loss
                losses += loss

                # Ajoutons l'accuracy
                accuracies += (predictions == labels).sum().item()

                if n_batches % 100 == 0:
                    self.writer.add_scalar(
                        "accuracy",
                        accuracies * 100 / (100 * self.batch_size),
                        global_step=n_batches,
                    )
                    self.writer.add_scalar(
                        "losses", losses / 100, global_step=n_batches
                    )
                    losses = 0
                    accuracies = 0

                n_batches += 1

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

        predictions = torch.round(outputs)

        print(predictions)
        # return [classes[prediction] for prediction in predictions.tolist()]
