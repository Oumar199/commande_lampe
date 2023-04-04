"""Ce module d'aide nous permettra de charger les échantillons (les signaux) ainsi que le taux d'échantillonnage 
"""
# from scipy.io import wavfile
import librosa
import librosa.display
import numpy as np
from glob import glob
import os

def extract_int_stft(tag_path: str, frame_size: int = 2048, hop_size: int = 512):
    """Cette fonction permet d'effectuer le stft (La transformation de fourier à court terme) et de retourner la matrice des fréquences par fenêtre
    ainsi que le taux d'échantillonnage

    Args:
        tag_path (str): Chemin vers le répertoire contenant les enregistrements audio
        frame_size (int): nombre d'échantillons par fenêtre. Defaults to 2048.
        hop_size (int): nombre d'échantillons de décalage. Defaults to 512.
    """
    for record_path in glob(os.path.join(tag_path, "*")):
        try:
            # Chargement des échantillons de l'actuel fichier audio
            y, sr = librosa.load(record_path)
            
            # whale_song, _ = librosa.effects.trim(y)
            
            # Initialisation du nombre d'échantillons par fenêtre
            FRAME_SIZE = frame_size
            
            # Initialisation du décalage 
            HOP_SIZE = hop_size
            
            # Effectuons le stft et récupérons les résultats en entiers et non en nombres complexes
            result = np.abs(librosa.stft(y, n_fft=FRAME_SIZE,  hop_length=HOP_SIZE))**2
            yield sr, result
            
        except Exception as e:
            print(f"error for {tag_path}: {e}")
            pass
