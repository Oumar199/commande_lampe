"""
Commande lampe version 4
---------------------------
Ce module doit nous permettre de prédire la classe d'une commande vocale avec le modèle de commande de lampe
"""
from speech_google.tests.utils.models.predict import predict_record
from speech_google.modelization.training.simple_run_training import SpecRunner3

def predict(key: str):
    """Effectuer une prédiction avec la version 4 du modèle de commande vocale de lampe

    Args:
        key (str): La clé permettant de récupérer le spectrogramme

    Returns:
        Tuple[str, torch.Tensor]: La classe prédite et sa probabilité d'apparition
    """
    # Initialisation la classe d'entraînement
    lampe_model = SpecRunner3(
        None,
        version = 4,
        means=[0.4654483497142792, 0.20280373096466064, 0.3366987705230713],
        stds=[0.384673535823822, 0.22996403276920319, 0.20797881484031677]
        )

    # Effectuons la compilation du modèle
    lampe_model.compile(output_size = 3, num_units=128, drop_out_rate=0.1, learning_rate=0.001)

    return predict_record(key, lampe_model, ['noth', 'off', 'on'], 'speech_google/modelization/storages/lampe_test3/')
