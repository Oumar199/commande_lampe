from speech_google.modelization.models.spec_models import SpecModel1
from torch import nn
import torch


def modify_multi_to_binary(model: nn.Module = SpecModel1()):
    """Fonction qui permet de transformer un modèle multi-classe en un modèle à sortie binaire

    Args:
        model (nn.Module, optional): Le modèle à modifié. Defaults to SpecModel1().
    """
    # Récupération de la dernière couche
    lin_layer = model.fc

    # On récupère le nombre d'entrées de la dernière couche
    in_feat = lin_layer.in_features # type: ignore

    # On modifie la dernière couche en fournissant une sigmoide en plus
    model.fc = nn.Sequential(nn.Linear(in_feat, 1), nn.Sigmoid()) # type: ignore

    return model
