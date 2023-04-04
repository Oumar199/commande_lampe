"""Ce module nous permet de sauvegarder les fichiers audio sous forme de spectrogrammes
"""
from speech_google.tests.utils.get_samples import extract_int_stft
from speech_google.tests.utils.get_tags import get_tags_from_dir
import librosa
import librosa.display
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import os

def create_spects(general_record_path: str = "speech_google/data/records/discussion/", spectrograms_path: str = "speech_google/data/spectrograms/discussion/", frame_size: int = 2048, hop_size: int = 512, replace = True, y_axis: str = "log"):
    """Créer et sauver les spectrogrammes

    Args:
        general_record_path (str, optional): Le chemin vers le dossier contenant les enregistrements vocals. Defaults to "speech_google/data/records/discussion/".
        spectrograms_path (str, optional): Le chemin vers le dossier devant contenir les spectrogrammes. Defaults to "speech_google/data/spectrograms/discussion/".
        frame_size (int): nombre d'échantillons par fenêtre. Defaults to 2048.
        hop_size (int): nombre d'échantillons de décalage. Defaults to 512.
        replace (bool): Indique s'il faut remplacer les images par de nouveaux. Defaults to True.
        y_axis (str): Spécifie l'échelle de l'axe des ordonnées. Defaults to 'log'.
    """
    # récupérons les chemins vers les étiquettes
    tag_paths = get_tags_from_dir(general_record_path)
    #print(len(tag_paths))
    if os.path.exists(spectrograms_path):
        for tag_path in tag_paths:
            
            label = os.path.basename(tag_path)
            
            #if label != "on" and label != "off":
            single_path = os.path.join(spectrograms_path, label)
            
            # initialisons notre générateur d'échantillons stft en type entier 
            gen_stft = extract_int_stft(tag_path, frame_size=frame_size, hop_size=hop_size)
            
            if replace:
                if os.path.exists(single_path):
                    for file_path in glob(os.path.join(single_path, "*")):
                        os.remove(file_path)
            
            for i, (sr, result) in enumerate(gen_stft):
                
                # Appliquons le logarithme sur les amplitudes du spectrogramme 
                Y_spec = librosa.power_to_db(result)
                
                # Tracons le spectrogramme
                librosa.display.specshow(Y_spec, sr = sr, hop_length = hop_size, x_axis = "time", y_axis = y_axis)
                
                # Enlevons les axes qui peuvent biaisés les résultats
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                
                if not os.path.exists(single_path):
                    os.makedirs(single_path)    
                    
                    # sauvegardons l'image sans bordure
                    plt.savefig(os.path.join(single_path, "1.png"), bbox_inches = "tight", pad_inches = 0)
                
                else:
                        
                    len_dir = len(os.listdir(single_path))

                    # sauvegardons l'image sans bordure
                    plt.savefig(os.path.join(single_path, f"{len_dir + 1}.png"), bbox_inches = "tight", pad_inches = 0)
                                
    else:
        raise OSError("Le fichier de sauvegarde des spectogrammes est introuvable !")
