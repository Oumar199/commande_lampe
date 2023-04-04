"""
Sauvegarde d'un enregistrement audio
-----------------------
Nous devrons sauvegarder une phrase ou une commande pour l'utiliser plus tard
"""

# importation de quelques librairies
from speech_google.tests.utils.error_handling import ignore_stderr
import pyaudio
import wave
import time
import os

# Let's use beepy module to make a coin sound before recording
try:
    from beepy import beep
    print_beep = False
except Exception:
	print_beep = True
	
class VoiceRecord(object):
    def __init__(self, n_chunks: int = 1024, sample_format: int = pyaudio.paInt16, channels: int = 2, frequency: int = 44100, seconds: int = 10):
        """Initialisation des principaux attributs

        Args:
            n_chunks (int, optional): Le nombre de troncons. Defaults to 1024.
            sample_format (int, optional): Le format des echantillons. Defaults to pyaudio.paInt16.
            channels (int, optional): Le nombre de chaines. Defaults to 2.
            frequency (int, optional): La frequence d'enregistrement. Defaults to 44100.
            seconds (int, optional): Le nombre de secondes. Defaults to 10.
        """
        self.n_chunks = n_chunks
        self.sample_format = sample_format
        self.channels = channels
        self.frequency = frequency
        self.seconds = seconds
    
    def led_alert(self, mode = "on"):
        """Permettre à l'utilisateur de connaître le début et la fin 
        d'un enregistrement
        
        Args: 
            mode (boot, optional): Indique le mode d'alerte, "on" pour allumer le led
            et "off" pour eteindre le led. Defaults to "on".
        """
        if mode == "on":
            # activons le buzzer
            with open("speech_google/data/node_interactions/buzzer_control.txt", "w") as f:
                f.write("31\n")
            time.sleep(1)
            
            # desactivons le buzzer
            with open("speech_google/data/node_interactions/buzzer_control.txt", "w") as f:
                f.write("32\n")
            
            # allumer le led
            with open("speech_google/data/node_interactions/ledrupt1.txt", "w") as f:
                    f.write("11\n")
            time.sleep(2.5)
            
            
        
        elif mode == "off":
            # eteindre le led
            with open("speech_google/data/node_interactions/ledrupt1.txt", "w") as f:
                    f.write("12\n")
        else:
            raise ValueError("Le mode spécifié n'est pas autorisé")
            
    
    def make_record(self, input: bool = True):
        """Enregistrement de voix

        Args:
            input (bool, optional): Mets l'enregistrement en mode entrée. Defaults to True.
        """
        with ignore_stderr():
            self.p = pyaudio.PyAudio()
			
        # we will use both of light alert and sound alert (with beep function and
        # electronic buzzer
        self.led_alert("on")
        
        if print_beep: print('\a')
        else: beep(sound = "coin")
		
        print("Début enregistrement ...")
        stream =  self.p.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.frequency,
            frames_per_buffer=self.n_chunks,
            input=input)
        
        self.frames = []
        
        for i in range(0, int(self.frequency / self.n_chunks * self.seconds)):
            data = stream.read(self.n_chunks)
            self.frames.append(data)
        self.led_alert("off")
        stream.stop_stream()
        stream.close()
        
        print("... Fin enregistrement")
        self.p.terminate()
        
        
        
        
    def save_last_record(self, path: str, filename: str = "output.wav"):
        """Sauvegardons le dernier enregistrement

        Args:
            path (str): Le chemin vers le répertoire de sauvegarde
            filename (str): Le nom du fichier de sauvegarde
        """
        if os.path.exists(path):
            ext = os.path.splitext(filename)[1]
            if ext == '.wav':
                    
                with wave.open(os.path.join(path, filename), "wb") as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.p.get_sample_size(self.sample_format))
                    wf.setframerate(self.frequency)
                    wf.writeframes(b''.join(self.frames))
            else:
                raise Exception(f"L'extension {ext} ne peut-être utilisée pour la sauvegarde !")
        else:
            raise Exception("Le chemin spécifié est introuvable !")
            
