"""
Sauvegarde d'un enregistrement audio
-----------------------
Nous devrons sauvegarder une phrase ou une commande pour l'utiliser plus tard
"""

# importation de quelques librairies
import pyaudio
import wave
import os

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
    
    def make_record(self, input: bool = True):
        """Record your voice

        Args:
            input (bool, optional): Make us to input voice record. Defaults to True.
        """
        self.p = pyaudio.PyAudio()
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
                raise Exception(f"The specified {ext} can not be used !")
        else:
            raise Exception("The specified path does not exist !")
            
