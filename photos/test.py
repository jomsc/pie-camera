# ------------------------------------------------------------------------------
# CODE POUR ENREGISTRER DES IMAGES AVEC LA CAMERA, A LANCER SUR LE RASPI
# ENREGISTRE LES IMAGES DANS YYYY-MM-DD/i.jpg
# ------------------------------------------------------------------------------

from picamera2 import Picamera2
import numpy as np
import time
from datetime import date

today = str(date.today())

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, 
                                                  lores={"size": (640,480)}, 
                                                  display="lores")
picam2.configure(camera_config)
picam2.start()

quit = False
i = 0
while not(quit):
    answer = input("do you want to take a new picture ? (y/n)")
    if answer=='y':
        time.sleep(2)
        picam2.capture_file(today+"/"+str(i)+".jpg")
        i += 1
    elif answer=='n':
        quit = True
        break
    else:
        print("incorrect answer.")
