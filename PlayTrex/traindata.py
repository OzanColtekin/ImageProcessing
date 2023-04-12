import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D , MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

imgs = glob.glob("./PlayTrex/img/*.png")

width = 125
height = 50

X = []
Y = []

for img in imgs:
    filename = os.path.basename(img)
    label = filename.split("_")[0]
    # read image
    im = np.array(Image.open(img).convert('L').resize((width,height)))

    # normalize data
    im = im / 255

    # data
    X.append(im)
    Y.append(label)

X = np.array(X)
X = X.reshape(X.shape[0], width, height, 1)


# encoding
def onehot_label(values):
    le = LabelEncoder()
    le_encoded = le.fit_transform(values)
    oe = OneHotEncoder(sparse=False)
    le_encoded = le_encoded.reshape(len(le_encoded),1)
    oe_encoded = oe.fit_transform(le_encoded)
    return oe_encoded

    
Y = onehot_label(Y)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=2)


# create model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape = (width,height,1)))
model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(3,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="Adam", metrics=["accuracy"])
model.fit(x_train,y_train,epochs=35,batch_size=64)

score_train = model.evaluate(x_train,y_train)
print(f"Eğitim Doğruluk: {score_train[1]*100}")

score_test = model.evaluate(x_test,y_test)
print(f"Test Doğruluk: {score_test[1]*100}")

open("model.json", "w").write(model.to_json())
model.save_weights("trex_weight.h5")