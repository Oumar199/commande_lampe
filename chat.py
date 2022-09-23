from speech_google.chats.lampe_chat import lampe_chat1, vocal_print

# def lampe_commande():
while True:
    quit_ = input("! Pour quitter : appuyer sur q\nSinon appuyer sur n'importe quel bouton ")
    response = ""
    if quit_ == "q":
        response = lampe_chat1(True)
        vocal_print(response)
        break
    response =lampe_chat1()
    vocal_print(response)