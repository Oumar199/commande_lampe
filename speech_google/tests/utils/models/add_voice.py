"""
Enregistrement momentané de voix
--------------------------------
Création d'une fonction qui permet d'enregistrer, au moment présent, notre voix 
"""
from speech_google.tests.utils.save_record import VoiceRecord
import string
import secrets
import winsound
frequency = 1500
duration = 500

def auto_voice_record(seconds: int = 4):
    """Enregistrement de voix et sauvegarde dans un fichier local 

    Args:
        seconds (int, optional): Le nombre de secondes nécessaire à l'enregistrement vocal. Defaults to 4.

    Returns:
        str: La clé générée pour le nom du fichier
    """
    # Génération d'un nom secret pour le fichier audio
    alphabet = string.ascii_letters + string.digits
    name = ''.join(secrets.choice(alphabet) for i in range(8))
    
    # Initialisation de la classe qui permet d'enregistrer des voix
    voice_record = VoiceRecord(seconds=seconds)
    
    # Enregistrement de la voix
    winsound.Beep(frequency, duration)
    voice_record.make_record()
    
    # Sauvegarde de l'enregistrement
    voice_record.save_last_record('./', f"{name}_voice.wav")
    
    # Retournons la clé
    return name
