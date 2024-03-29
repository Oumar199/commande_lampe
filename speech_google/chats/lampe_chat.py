"""
Fonction lampe-chatbot 
-----------------
Ce module doit nous permettre de faire parler un bot en réponse à la commande vocale donnée pour le modèle
de commande vocale de lampe. 

Copyright © 2022 OkEngineering 
"""
from speech_google.tests.utils.models.add_spectrogram import auto_create_spec
from speech_google.tests.lampe_predict1 import predict, predict2
from gtts import gTTS
import random
import pyglet
import torch
import json
import time
import os

# Chargement du fichier json qui nous concerne
with open("speech_google/data/json/lampe_test1.json", "r") as f:
    lampe_json = json.load(f)


def vocal_print(text: str):
    """Cette fonction permettra au bot de parler

    Args:
        text (str): Le texte qui doit être dit
    """
    try:
        # Nous allons utiliser la langue francaise
        speech = gTTS(text, lang="fr")

        # Sauvegarde temporaire dans un fichier mp3
        file_path = "voice.mp3"
        speech.save(file_path)

        # Chargement et démarrage de l'audio
        voice = pyglet.media.load(file_path, streaming=False)
        voice.play()

        # Temps de latence avant la prochaine commande vocale
        time.sleep(voice.duration + 4)

        # Suppression du fichier mp3
        os.remove(file_path)
    except AssertionError:
        pass


def lampe_chat1(exit_: bool = False, bot_name: str = "Elisa", thres: float = 0.7):
    """Effectuer une échange avec le bot.

    Args:
        exit_ (bool, optional): Indiquer si l'utilisateur souhaite quitter. Defaults to False.
        bot_name (str, optional): Le nom du bot. Defaults to "Elisa".
        thres (float, optional): Le seuil de probabilité minimale de la classe. Defaults to 0.7.

    Returns:
        str: La réponse fournie par le bot
    """
    if exit_ == False:
        # Enregistrement d'une commande
<<<<<<< HEAD
        key = auto_create_spec(dpi = 100)
=======
        key = auto_create_spec()
>>>>>>> 6da7ce28ee5ee7a91878cb6c1a0429d5b95622c0
        time.sleep(5)
        # Récupération de la prédiction et de sa probabilité
        # class_name, prob = predict(key) # we will use the second function
        class_name, prob = predict(key)
        
<<<<<<< HEAD
        #prob = prob.item()  # On récupère la probabilité en nombre et non en tensor
        
        #---------------------
        #prob = 0.5
        #class_name == "noth"
        #---------------------
        
=======
>>>>>>> 6da7ce28ee5ee7a91878cb6c1a0429d5b95622c0
        print(class_name, prob)
        # Initialisation de la réponse à ""
        response = ""
        # Le bot ne donne une réponse que si la probabilité de la classe prédite dépasse un seuil fourni
        if class_name == "noth":
                pass
        else:
            if prob > thres:
                if class_name == "on":
                    option = "101"
                elif class_name == "off":
                    option = "102"

                with open('speech_google/data/node_interactions/led_control.txt', 'w') as f:
                    f.write(option)
                # On récupère aléatoirement une réponse et on remplace 'bot_name', s'il y figure, par le nom du bot
                for intent in lampe_json["lampe_test1"]:
                    if class_name == intent["tag"]:
                        response = f"{random.choice(intent['responses']).replace('bot_name', bot_name)}"
            else:
                response = "Je n'ai pas compris ce que vous aviez dit ! Veuillez répéter s'il vous plaît"
        return response
    else:
        return "Au revoir. J'espère qu'on se reverra bientôt"
