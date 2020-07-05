import tensorflow as tf
from tensorflow import keras
import pandas as pd 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

import time

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

#path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

#PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
PATH = '/.keras/datasets/cats_and_dogs_filtered'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures

validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_test_tr = len(os.listdir(test_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
print("Total test images:", num_test_tr)

batch_size = 128
epochs = 10
IMG_HEIGHT = 80
IMG_WIDTH = 80
Image_Channels=3

t1 = time.time()

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')                                                  


test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')  


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,Image_Channels)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


t2 = time.time()
print('[INFO] compile')
#model.summary()

print("[INFO] fit...")

model.fit(train_data_gen,epochs=epochs, batch_size=32)

t3 = time.time()
print("[INFO] predict...")

pred = model.predict(test_data_gen)

t4 = time.time()

predicted_val = [int(round(p[0])) for p in pred]

pred_arr = []
for i, p in enumerate(pred):
    print('i: ' + str(i))
    if p > 0.5:
        print("%.2f" % (p[0]*100) + "% cat")
        predicted_val[i] = 0
    else:
        print("%.2f" % ((1-p[0])*100) + "% dog")
        predicted_val[i] = 1

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_data_gen.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_val]

filenames=test_data_gen.filenames

for i, s in enumerate(filenames):
    filenames[i] = os.path.basename(s.strip()) 


results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results11.csv",index=False)


print('time compile: ' + str(t2-t1) + ' seconds')
print('time fit: ' + str(t3-t2) + ' seconds')
print('time predict: ' + str(t4-t3) + ' seconds')


print('end')