import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


no_of_epoches = 30
#image_X = 320
#image_Y = 240
image_X = 160
image_Y = 120

tr_tt_ratio = 0.8
F0 = 'G:\\T\\G0'
F1 = 'G:\\T\\G1'
n0 = 0
n1 = 0
for filename in os.listdir(F0):
    n0 = n0 + 1
for filename in os.listdir(F1):
    n1 = n1 + 1
  
n = n0 + n1
    
im = np.empty((n,image_Y,image_X,3))
cl = np.empty(n)
cl = cl.astype(int)

i = 0
for filename in os.listdir(F0):    
    fn1 = os.path.join(F0,filename)
    temp1 = cv2.imread(fn1)
    tempo= cv2.resize(temp1,dsize=(image_X,image_Y),interpolation=cv2.INTER_CUBIC)
    tempo = tempo/255.0
    im[i] = tempo
    cl[i] = 1
    i = i + 1

i = n0
for filename in os.listdir(F1):    
    fn1 = os.path.join(F1,filename)
    temp1 = cv2.imread(fn1)
    tempo= cv2.resize(temp1,dsize=(image_X,image_Y),interpolation=cv2.INTER_CUBIC)
    tempo = tempo/255.0
    im[i] = tempo
    cl[i] = 0
    i = i + 1

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(im[i])
plt.show()

ntr = round(n*tr_tt_ratio)
ntt = n - ntr
tri = [random.randint(1,n) for i in range(0,ntr)]
tot = np.arange(n)
tes = np.setdiff1d(tot,tri)

train_images, test_images, train_labels, test_labels = train_test_split(im, cl, test_size=1-tr_tt_ratio, random_state=0)

i = 0
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(train_labels[i])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_Y, image_X, 3)))
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2))
model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=no_of_epoches, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
