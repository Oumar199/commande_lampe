"""Ce module va nous aider à charger les chemins vers les différentes étiquettes du dossier de sauvegarde
"""
from glob import glob
import os

def get_tags_from_dir(path:str = "speech_google/data/records/discussion/"):
    """Récupération des chemins étiquettes à partir du dossier de sauvegarde

    Args:
        path (str, optional): Le chemin vers le dossier de sauvegarde. Defaults to "speech_google/data/records/discussion".
    """
    tag_paths = []
    if os.path.exists(path):
        for tag_path in glob(os.path.join(path, "*")):
            if os.path.splitext(tag_path)[1] == "":
                tag_paths.append(tag_path)
    else:
        print("Le chemin spécifié est introuvable. Veuilliez réessayer svp.")
    return tag_paths
            
            
    
