import keyboard
import uuid
import time
import os
from PIL import Image
from mss import mss

# frame coordinats
monitor = {"top": 510, "left":580, "width":250, "height":100}

sct = mss()

index = 0

def record_screen(record_id, key):
    global index
    index +=1
    print(f"{key}:{index}")
    img = sct.grab(monitor)
    im = Image.frombytes("RGB",img.size,img.rgb)
    im.save("./PlayTrex/img/{}_{}_{}.png".format(key,record_id,index))

is_exit = False

def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey("esc",exit)

record_id = uuid.uuid4()

while True:
    if is_exit : break

    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id,"up")
            time.sleep(0.1)
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id,"down")
            time.sleep(0.1)
        elif keyboard.is_pressed("right"):
            record_screen(record_id,"right")
            time.sleep(0.1)
    except RuntimeError:
        continue