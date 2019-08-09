# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing CNN libraries
import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten

# initializing CNN
classifier = Sequential()
#adding convolutional layer
classifier.add(Conv2D(32,(3,3), activation = "relu", input_shape = (32,32,3)))
# adding pooling layer
classifier.add(MaxPool2D(pool_size = (2,2)))
# adding another convolution layer
classifier.add(Conv2D(32,(3,3), activation = "relu"))
# adding another pooling layer
classifier.add(MaxPool2D(pool_size = (2,2)))
# flatten
classifier.add(Flatten())
# full connection
classifier.add(Dense(units = 26, activation = 'softmax'))
# compiling
classifier.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

# Image transformation on images
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

#Same for the test set as well
test_datagen = ImageDataGenerator(rescale=1./255)

#get the training data set using the flow from directory method
train_generator = train_datagen.flow_from_directory('training_set',
                                          target_size=(32, 32),
                                          batch_size=32,
                                          class_mode='categorical')

#Same for the test datasets
validation_generator = test_datagen.flow_from_directory('test_set',
                                        target_size=(32, 32),
                                        batch_size=32,
                                        class_mode='categorical')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=30,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=25)

# testing
from keras.preprocessing import image
test_image = image.load_img('im.jpg', target_size=(32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = classifier.predict(test_image)
validation_generator.class_indices

list1 = ['A','B','C','D','E','F',
         'G','H','I','J','K','L',
         'M','N','O','P','Q','R',
         'S','T','U','V','W','X']
for i in range(0,26):
    if (result[0][i] == 1):
        print('Letter is :',list1[i])
    
