"""Création d'un modèle plus simple
"""
import torch
from torch import nn
from torch.nn import functional as F

class SpecModel2(nn.Module):
    def __init__(self, drop_out_rate: float = 0.3, num_units: int = 400, output_size: int = 2):
        """Initialisation des couches

        Args:
            drop_out_rate (float): Le taux du drop out. Defaults to 0.3.
            num_units (int, optional): Nombre d'unités de la première couche dense. Defaults to 400.
            output_size (int, optional): Nombre de classes. Defaults to 2.
        """
        super(SpecModel2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 32, 6)
        self.fc1 = nn.Linear(32*51*80, num_units)
        self.drop = nn.Dropout(drop_out_rate)
        self.fc2 = nn.Linear(num_units, output_size)
        
    def forward(self, input: torch.Tensor):
        """Passage vers l'avant 

        Args:
            input (torch.Tensor): Les données d'entrée
        """
        out = self.pool(F.relu(self.conv1(input)))
        out = self.pool(F.relu(self.conv2(out)))
        out = out.view(-1, 32*51*80)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        return self.fc2(out)
