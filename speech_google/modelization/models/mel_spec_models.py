import torch
from torch import nn
from torch.nn import functional as F
from speech_google.modelization.models.utils.calculate_new_width import new_width_after_conv
"""Création du troisième modèle de notre projet  
"""

class SpecModel3(nn.Module):
    def __init__(self, input_shape: tuple = (308, 465), o_channel: tuple = (16, 32), drop_out_rate: float = 0.3, num_units: int = 400, output_size: int = 2, seed: int = 100):
        """Initialisation des couches

        Args:
            input_shape (tuple): Un tuple contenant les dimensions (hauteur et largeur) des données d'entrées. Defaults to (308, 465).
            o_channel (tuple): Un tuple indiquant le nombre de chaines de sortie pour chaque couche de convolution. Defaults to (16, 32).
            f_size1 (int): La taille du filtre de la première couche de convolution. Defaults to 
            drop_out_rate (float): Le taux du drop out. Defaults to 0.3.
            num_units (int, optional): Nombre d'unités de la première couche dense. Defaults to 400.
            output_size (int, optional): Nombre de classes. Defaults to 2.
            seed (int): Graine du generateur. Defaults to 100.
        """
        super(SpecModel3, self).__init__()
        if seed: torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, o_channel[0], 16, stride = 2) # new_shape = (32, 147, 225)
        
        input_shape = (new_width_after_conv(input_shape[0], 16, 0, 0, 2, False),
                     new_width_after_conv(input_shape[1], 16, 0, 0, 2, False))
        
        self.pool1 = nn.MaxPool2d(2, 2) # new_shape = (32, 73, 112)
        
        input_shape = (new_width_after_conv(input_shape[0], 2, 0, 0, 2, False),
                     new_width_after_conv(input_shape[1], 2, 0, 0, 2, False))
        
        self.conv2 = nn.Conv2d(o_channel[0], o_channel[1], 20, stride = 2) # new_shape = (64, 27, 47)
        
        input_shape = (new_width_after_conv(input_shape[0], 20, 0, 0, 2, False),
                     new_width_after_conv(input_shape[1], 20, 0, 0, 2, False))
        
        self.pool2 = nn.MaxPool2d(2, 2) # new_shape = (64, 13, 23)
        
        self.o_channel = o_channel
        self.input_shape = (new_width_after_conv(input_shape[0], 2, 0, 0, 2, False),
                     new_width_after_conv(input_shape[1], 2, 0, 0, 2, False))
        
        self.fc1 = nn.Linear(self.o_channel[1]*self.input_shape[0]*self.input_shape[1], num_units) 
        self.drop = nn.Dropout(drop_out_rate)
        self.fc2 = nn.Linear(num_units, output_size)
        
    def forward(self, data: torch.Tensor):
        """Passage vers l'avant 

        Args:
            input (torch.Tensor): Les données d'entrée
        """
        data = data.unsqueeze(0) 
        
        out = self.pool1(F.relu(self.conv1(data)))
        
        out = self.pool2(F.relu(self.conv2(out))) # new_shape = (64, 13, 23)
        
        out = out.view(-1, self.o_channel[1]*self.input_shape[0]*self.input_shape[1])
        
        out = F.relu(self.fc1(out))
        
        out = self.drop(out)
        
        return self.fc2(out)
        
