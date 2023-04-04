"""
Création automatique de spectrogramme
-----------------------------
Nous allons créer automatiquement un spectrogramme à partir d'un enregistrement vocal localement sauvegardé
"""
from speech_google.tests.utils.models.add_voice import auto_voice_record
import matplotlib.pyplot as plt
from typing import Union
import librosa.display
import numpy as np
import librosa
import os


def auto_create_spec(
    frame_size: int = 2048, hop_size: int = 512, seconds: int = 4, y_axis: str = "log", dpi: Union[int, None] = None
):
    """Enregistrement d'une voix et création d'un spectrogramme

    Args:
        frame_size (int, optional): Le nombre d'échantillons par fenêtre. Defaults to 2048.
        hop_size (int, optional): Le décalage de la fenêtre. Defaults to 512.
        seconds (int, optional): Le nombre de secondes par enregistrement. Defaults to 4.
        y_axis (str, optional): Transformation effectuée sur l'axe des ordonnées. Defaults to "log".
        dpi (Union[int, None]: Pixels par pouce. Defaults to None.
    """
    # Effectuons d'abord l'enregistrement vocal
    name = auto_voice_record(seconds=seconds)

    # Chargement des échantillons de l'actuel fichier audio
    y, sr = librosa.load(f"{name}_voice.wav")

    # Initialisation du nombre d'échantillons par fenêtre
    FRAME_SIZE = frame_size

    # Initialisation du décalage
    HOP_SIZE = hop_size

    # Effectuons le stft et récupérons les résultats en entiers et non en nombres complexes
    result = np.abs(librosa.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)) ** 2

    # Appliquons le logarithme sur les amplitudes du spectrogramme
    Y_spec = librosa.power_to_db(result)
    
    # Définition d'un figsize pour fixer la taille des spectrogrammes
    #plt.figure(figsize = (6*308, 4*496/465))
    plt.figure(figsize = (6*(496/465), 4*(369/308)))
    #plt.figure(figsize = (6, 4))
    
    # Tracons le spectrogramme
    librosa.display.specshow(
        Y_spec, sr=sr, hop_length=hop_size, x_axis="time", y_axis=y_axis
    )
    
    # Enlevons les axes qui peuvent biaisés les résultats
    ax = plt.gca()
    #plt.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Sauvegardons l'image sans bordure
    plt.savefig(f"{name}_spectrogram.png", bbox_inches="tight", pad_inches=0, dpi = dpi)

    # Suppression de l'enregistrement vocal
    os.remove(f"{name}_voice.wav")

    # Retournons la clé
    return name
