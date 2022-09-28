from speech_google.chats.lampe_chat import lampe_chat1, vocal_print
import datetime
import serial
import time
time_1 = datetime.datetime.now()
while True:
    try:
        ser = serial.Serial("/dev/ttyACM_OKARD1", 9600, timeout = 1)
        break
    except Exception:
        print("Check if the arduino is plugged !")
        pass
        

while True:
    try:
        with open("speech_google/data/node_interactions/ledrupt1.txt", "r") as f:
            i = f.readlines()[0].strip()
        
        with open("speech_google/data/node_interactions/ledrupt2.txt", "r") as f:
                j = f.readlines()[0].strip()
            
        with open("speech_google/data/node_interactions/dis1.txt", "r") as f:
            l = f.readlines()[0].strip()
        
        with open("/home/pi/Projects/commande_lampe/speech_google/data/node_interactions/chat_dis.txt", "r") as f:
            c = f.readlines()[0].strip()
        
        if c == "222":
            with open("speech_google/data/node_interactions/LEDrupt1.txt", "r") as f:
                k = f.readlines()[0].strip()
                with open('speech_google/data/node_interactions/led_control.txt', 'w') as f:
                    f.write(k)
        else:
            with open('speech_google/data/node_interactions/led_control.txt', 'r') as f:
                k = f.readlines()[0].strip()
        
        ser.write(f"{i},{j},{k},{l}".encode('utf-8'))
        
        time.sleep(0.5)
        
        print(f"{i},{j},{k},{l}")
    
        data = ser.readline().decode('utf-8').rstrip()
        data = data.split(',')
        
        if len(data) == 3:
            with open('speech_google/data/node_interactions/temp.txt', 'w') as f:
                f.write(data[0])
            with open('speech_google/data/node_interactions/humid.txt', 'w') as f:
                f.write(data[1])
            with open('speech_google/data/node_interactions/dist.txt', 'w') as f:
                f.write(data[2])
                
        response = "..."
        now_ = datetime.datetime.now()
        if c == "221" and (now_ - time_1).seconds >= 5:
            class_name, response = lampe_chat1()
            time_1 = datetime.datetime.now()
            #vocal_print(response)
            
            option = None
            if class_name == "on":
                option = "101"
            elif class_name == "off":
                option = "102"
            print(class_name)
            if option:
                with open('speech_google/data/node_interactions/led_control.txt', 'w') as f:
                    f.write(option)
        
        with open('speech_google/data/responses/response.txt', 'w') as f:
            f.write(response)

    except Exception as e:
        print(f"An error occured: {e}")
        try:
            ser = serial.Serial("/dev/ttyACM_OKARD1", 9600, timeout = 1)
        except Exception:
            pass
