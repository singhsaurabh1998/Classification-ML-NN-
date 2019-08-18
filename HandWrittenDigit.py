from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
'''mnist data set for hand written digits contains 70000 images ,each image contains 784(28x28) features'''
# importing dataset
mnist = fetch_openml('mnist_784')

x = mnist['data']  # contains the images as array
y = mnist['target']  # labels corresponding to the images

#plt.imshow(x[0].reshape(28,28)) # ploting the first image

# splitting into training & test set
x_train = x[:6000]
x_test = x[6000:]

y_train = y[:6000]
y_test = y[6000:]

# shuffling the data
shuffle_index = np.random.permutation(6000)
x_train = x_train[shuffle_index]
y_train = y_train[shuffle_index]

#  convert the string data into the digits
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

# training the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(tol=.2, multi_class='auto') # tolerance is as per ur choive for fast training
classifier.fit(x_train,y_train)

#prediction
classifier.predict([x_test[0]]) # must be same as the ytest

