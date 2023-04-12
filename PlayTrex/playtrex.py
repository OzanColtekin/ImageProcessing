from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss

# frame coordinats
monitor = {"top": 510, "left":580, "width":250, "height":100}
sct = mss()

width = 125
height = 50

# upload model
model = model_from_json(open('./PlayTrex/model.json',"r").read())
model.load_weights("./PlayTrex/trex_weight.h5")

labels = ["Down","Right","Up"]


frame_time = time.time()
counter = 0
i = 0
delay = 0.4
key_down_pressed = False

while True:
    img = sct.grab(monitor)
    im = Image.frombytes("RGB",img.size,img.rgb)
    im = np.array(im.convert('L').resize((width,height)))
    im = im / 255

    X = np.array([im])
    X = X.reshape(X.shape[0],width,height, 1)

    r = model.predict(X)
    result = np.argmax(r)

    if result == 0:
        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed = True
    elif result == 2:
        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
        time.sleep(delay)
        keyboard.press(keyboard.KEY_UP)
        if i < 1500 :
            time.sleep(0.3)
        elif 1500 < i and i < 5000:
            time.sleep(0.2)
        else:
            time.sleep(0.13)

        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)

    counter +=1
    if (time.time() - frame_time) > 1:
        counter = 0
        frame_time = time.time()
        if i < 1500 :
            delay -= 0.003
        else:
            delay -= 0.005
        if delay < 0:
            delay = 0

        i+=1
        print("-----------------")
        print(f"Down: {r[0][0]} \n Right: {r[0][1]} \n Up: {r[0][2]}")