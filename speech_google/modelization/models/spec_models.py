"""Ici sont définis les différentes modèles pour les spectogrammes
"""
import torch
from torch import nn
from torch.nn import functional as F

class SpecModel1(nn.Module):
    def __init__(self, drop_out_rate: float = 0.3, num_units1: int = 3000, num_units2: int = 1000, num_units3: int = 300, output_size: int = 7):
        """Initialisation des couches

        Args:
            drop_out_rate (float): Le taux du drop out. Defaults to 0.3.
            num_units1 (int, optional): Nombre d'unités de la première couche dense. Defaults to 3000.
            num_units2 (int, optional): Nombre d'unités de la deuxième couche dense. Defaults to 1000.
            num_units2 (int, optional): Nombre d'unités de la troisième couche dense. Defaults to 300.
            output_size (int, optional): Nombre d'unités de la quatrième couche dense. 7.
        """
        super(SpecModel1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 6)
        self.conv3 = nn.Conv2d(32, 64, 6)
        self.lin1 = nn.Linear(64*31*49, num_units1)
        self.lin2 = nn.Linear(num_units1, num_units2)
        self.drop = nn.Dropout(drop_out_rate)
        self.lin3 = nn.Linear(num_units2, num_units3)
        self.fc = nn.Linear(num_units3, output_size)
        
    def forward(self, input: torch.Tensor):
        """Passage vers l'avant 

        Args:
            input (torch.Tensor): Les données d'entrée
        """
        out = self.pool(F.relu(self.conv1(input)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))
        out = out.view(-1, 64*31*49)
        out = F.relu(self.lin1(out))
        out = F.relu(self.lin2(out))
        out = self.drop(out)
        out = F.relu(self.lin3(out))
        return self.fc(out)
