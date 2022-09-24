"""
Prédiction de la classe d'un enregistrement vocal
--------------------------
On suppose qu'un modèle prédisant la classe d'un enregistrement vocal a dèja été créé et qu'un spectrogramme est prêt 
à l'emploi.
"""
from speech_google.modelization.training.run_training import SpecRunner1
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from typing import Union, List
import PIL.Image as pi
import torch
import os


def predict_record(
    key: str,
    runner: SpecRunner1,
    classes: List[Union[str, int]],
    model_path: str = "speech_google/modelization/storages/",
):
    """Récupérer et prédire la classe d'un enregistrement à l'aide d'un modèle pré-entraîné et d'un spectrogramme

    Args:
        key (str): La clé du fichier
        runner (SpecRunner1): La classe d'entraînement qui va nous permettre d'effectuer la prédiction.
        classes (List[Union[str, int]]): Les différentes classes possibles réunies dans une liste.
        model_path (str): Le chemin vers le fichier
    """
    path = f"{key}_spectrogram.png"
    if os.path.exists(path):
        # Récupérons l'image
        image = pi.open(path).convert("RGB") # type: ignore

        # Initialisation du transformateur
        compose = Compose([ ToTensor(), Resize((220, 338)), Normalize(runner.means, runner.stds)])
        # compose = Compose([ToTensor(), Normalize(runner.means, runner.stds)])

        # Effectuons de la transformation sur l'image
        image: torch.Tensor = compose(image)

        try:
            # Chargeons le modèle
            runner.load_model(saving_path=model_path)

            with torch.no_grad():
                # Mettons le modèle en mode évaluation
                runner.model.eval()

                # Récupérons la sortie du modèle après avoir fourni les données
                output = runner.model(image)

                # Effectuons une prédiction
                _, prediction = torch.max(output.data, dim=1)

                # Récupérons le nom de la classe
                class_name = classes[prediction.item()]

                # Récupérons les probabilités
                probs = torch.softmax(output, dim=1)

                # Récupérons la probabilité de la classe prédite
                prob = probs[0][prediction.item()]

            # Suppression du spectrogramme
            #os.remove(path)

            # Retournons le nom de la classe prédite ainsi que sa probabilité
            return class_name, prob

        except Exception as e:
            raise Exception("Une erreur s'est produite !")

    else:
        raise ValueError("L'enregistrement vocal est indisponible")
