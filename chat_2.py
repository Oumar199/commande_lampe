import concurrent.futures
from speech_google.chats.lampe_chat import lampe_chat1, vocal_print
import datetime
import serial
from glob import glob
import time
import os

c = None

        
        
def multi_processes(process):
    global c
    time_1 = datetime.datetime.now()

    if process == "data":
        while True:
            try:
                with open("speech_google/data/node_interactions/ledrupt1.txt", "r") as f:
                    i = f.readlines()[0].strip()
                    #with open('speech_google/data/node_interactions/led1_control.txt', 'w') as f:
                    #    f.write(i)
				
                with open("speech_google/data/node_interactions/ledrupt2.txt", "r") as f:
                    j = f.readlines()[0].strip()
                    with open('speech_google/data/node_interactions/led2_control.txt', 'w') as f:
                        f.write(j)
				
<<<<<<< HEAD
                with open("speech_google/data/node_interactions/chat_dis.txt", "r") as f:
=======
                with open("/home/pi/Projects/commande_lampe/speech_google/data/node_interactions/chat_dis.txt", "r") as f:
>>>>>>> 6da7ce28ee5ee7a91878cb6c1a0429d5b95622c0
                    c = f.readlines()[0].strip()
                    with open('speech_google/data/node_interactions/chat_en_control.txt', 'w') as f:
                        f.write(c)
				
                with open("speech_google/data/node_interactions/dis1.txt", "r") as f:
                    l = f.readlines()[0].strip()
                
                with open("speech_google/data/node_interactions/buzzer_control.txt", "r") as f:
                    b = f.readlines()[0].strip()
				
                if c == "222":
                    with open("speech_google/data/node_interactions/LEDrupt1.txt", "r") as f:
                        k = f.readlines()[0].strip()
                        with open('speech_google/data/node_interactions/led_control.txt', 'w') as f:
                            f.write(k)
					
                else:
                    with open('speech_google/data/node_interactions/led_control.txt', 'r') as f:
                        k = f.readlines()[0].strip()
				
                ser.write(f"{i},{j},{k},{l},{b}".encode('utf-8'))
				
                time.sleep(0.5)
				
                #print(f"{i},{j},{k},{l},{b}")
			
                data = ser.readline().decode('utf-8').rstrip()
                data = data.split(',')
				
                if len(data) == 3:
                    with open('speech_google/data/node_interactions/temp.txt', 'w') as f:
                        f.write(data[0])
                    with open('speech_google/data/node_interactions/humid.txt', 'w') as f:
                        f.write(data[1])
                    with open('speech_google/data/node_interactions/dist.txt', 'w') as f:
                        f.write(data[2])
						
            except Exception as e:
                print(f"An error occured: {e}")
                try:
                    ser = serial.Serial("/dev/ttyACM_OKARD1", 9600, timeout = 1)
                except Exception:
                    pass
					
    elif process == "chat":
        while True:
            try:
                now_ = datetime.datetime.now()
                if c == "221" and (now_ - time_1).seconds >= 5:
                    response = lampe_chat1()
                    time_1 = datetime.datetime.now()
                    #vocal_print(response)
					
                    with open('speech_google/data/responses/response.txt', 'w') as f:
                        f.write(response)
            
            except Exception as e:
                print(f"An error occured: {e}")
				
def main():
    #while True:
    #    try:
    #        ser = serial.Serial("/dev/ttyACM_OKARD1", 9600, timeout = 1)
    #        break
    #    except Exception:
    #        print("Check if the arduino is plugged !")
    #        pass
    for file_path in glob("./*.wav"):
        os.remove(file_path)
    for file_path in glob("./*.png"):
        os.remove(file_path)
    with open("speech_google/data/node_interactions/ledrupt1.txt", "w") as f:
        f.write("12/n");
    with open("speech_google/data/node_interactions/buzzer_control.txt", "w") as f:
        f.write("32/n");
 
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(multi_processes, ["data", "chat"])

if __name__ == "__main__":
	main()	
