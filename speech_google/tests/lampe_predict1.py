"""
Commande lampe version 4
---------------------------
Ce module doit nous permettre de prédire la classe d'une commande vocale avec le modèle de commande de lampe
"""
#version 8
#means = [0.4332595467567444, 0.1864638477563858, 0.34693169593811035],
#stds = [0.3609752655029297, 0.211887925863266, 0.22041478753089905]
from speech_google.tests.utils.models.predict import predict_record
from speech_google.modelization.training.simple_run_training import SpecRunner3
from speech_google.modelization.training.mel_run_training import SpecRunner4


def predict(key: str):
    """Effectuer une prédiction avec la version 4 du modèle de commande vocale de lampe

    Args:
        key (str): La clé permettant de récupérer le spectrogramme

    Returns:
        Tuple[str, torch.Tensor]: La classe prédite et sa probabilité d'apparition
    """
    # Initialisation la classe d'entraînement
    lampe_model = SpecRunner3(
        None, # type: ignore
        version=12,
        means=[0.34141504764556885, 0.11615464836359024, 0.32915809750556946],
        stds=[0.306147038936615, 0.13027192652225494, 0.19265204668045044]
    )

    # Effectuons la compilation du modèle
    lampe_model.compile(
        output_size=3, num_units=128, drop_out_rate=0.2, learning_rate=0.0001
    )

    return predict_record(
        key,
        lampe_model,
        ["noth", "off", "on"],
        "speech_google/modelization/storages/lampe_test3/",
    )

def predict2(key: str):
    """Effectuer une prédiction avec la version 4 du modèle de commande vocale de lampe

    Args:
        key (str): La clé permettant de récupérer le spectrogramme

    Returns:
        Tuple[str, torch.Tensor]: La classe prédite et sa probabilité d'apparition
    """
    # Initialisation la classe d'entraînement
    lampe_model = SpecRunner4(
        None, # type: ignore
        version=11,
        means=[0.3589266240596771, 0.1221177726984024, 0.34370312094688416],
        stds=[0.3050106167793274, 0.13116586208343506, 0.1847974807024002]
    )

    # Effectuons la compilation du modèle
    lampe_model.compile(
        input_shape = (369, 496), o_channel = (32, 64), drop_out_rate=0.1, num_units=128, output_size=3, learning_rate=0.00001
    )
    
    return predict_record(
        key,
        lampe_model,
        ["noth", "off", "on"],
        "speech_google/modelization/storages/lampe_test3/",
    )
