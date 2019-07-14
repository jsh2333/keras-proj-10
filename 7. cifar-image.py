# CIFAR10 Classification with Keras CNN
# July 8, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# imports
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers
from keras import datasets
from keras.utils import np_utils
from keras.models import Sequential


# global constants and hyper-parameters
NUM_CLASS = 10
MY_EPOCH = 10
MY_BATCH = 64
MY_VALID = 0.2
MY_SAMPLE = 52


    ####################
    # DATABASE SETTING #
    ####################


# read the DB from keras
(X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()


# print shape information
def show_shape():
    print('\n== DB SHAPE INFO ==')
    print('X_train shape = ', X_train.shape)
    print('X_test shape = ', X_test.shape)
    print('Y_train shape = ', Y_train.shape)
    print('Y_test shape = ', Y_test.shape)    
    print()

show_shape()


# define labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck']


# show a sample image and its label
# note that Y_train array is 2-dimensional
# we need to access the first axis to get the label
print('\n== SAMPLE DATA (RAW) ==')
sample = Y_train[MY_SAMPLE, 0]
print("This is a", labels[sample], sample)
print(X_train[MY_SAMPLE])
plt.imshow(X_train[MY_SAMPLE])
plt.show()


# reshaping is not necessary
# the data already has (L, W, H, C) shape that CNN needs
# the channel info has 3 RGB channels
# we convert integer gray scale into float (data scaling)
X_train = X_train / 255.0
X_test = X_test / 255.0


# one-hot encoding of the outputs
# convert [0, 9] decimal label into an array of 10 binary data
# useful to calculate categorical_crossentropy
Y_train = np_utils.to_categorical(Y_train, NUM_CLASS)
Y_test = np_utils.to_categorical(Y_test, NUM_CLASS)
show_shape()

print('\n== SAMPLE DATA (PROCESSED)==')
print("This is a", Y_train[MY_SAMPLE])
print(X_train[MY_SAMPLE])


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# keras sequential model for CNN
# total parameter count formula =
# (filter_height * filter_width * input_channels + 1) * #_filters
# in order to handle multi-channel inputs, the filter becomes 3D
# here, input_channel = filter_thickness
model = Sequential()
shape = (32, 32, 3)

# input layer with (32 x 32) image size & 3 RGB channels
model.add(layers.InputLayer(input_shape = shape))

# first convolution with 8 3x3 filters
# no change in image size: (32 x 32)
# number of images: 8
# dropout rate is 0.25
# no. parameters = (3 x 3 x 3 + 1) x 8
model.add(layers.Conv2D(8, (3, 3), padding = 'same', activation = 'relu'))
model.add(layers.Dropout(0.25))

# second convolution with 16 3x3 filters
# no change in image size: (32 x 32)
# number of images: 16
# no. parameters = (3 x 3 x 8 + 1) x 16
model.add(layers.Conv2D(16, (3, 3), padding = 'same', activation = 'relu'))

# first max pooling
# image size reduces to (16 x 16)
# number of images: 16
model.add(layers.MaxPooling2D(pool_size = (2, 2)))

# getting ready for fully-connected layer  
# image flattened to 1D vector with 4096 (= 16 x 16 x 16) pixels
# dropout rate is 0.35
model.add(layers.Flatten())
model.add(layers.Dropout(0.35))

# form 4096 x 128 fully-connected layer 
# dropout rate is 0.5
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

# form 128 x 10 fully-connected layer 
# final output with 10 classes
model.add(layers.Dense(NUM_CLASS, activation = 'softmax'))
model.summary()


# testing the model before learning with a sample
sample = X_train[MY_SAMPLE]
sample = sample.reshape(1, 32, 32, 3)
pred = model.predict(sample)
print("\nPrediction before learning:", labels[np.argmax(pred)], "\n")


# model training and saving
model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', 
        metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = MY_EPOCH, batch_size = MY_BATCH, 
        validation_split = MY_VALID)
model.save('chap7.h5')


    ####################
    # MODEL EVALUATION #
    ####################


# conduct evaluation
score = model.evaluate(X_test, Y_test, verbose = 1)
print('\nKeras CNN model loss = ', score[0])
print('Keras CNN model accuracy = ', score[1])


# testing the model after learning with a sample
sample = X_train[MY_SAMPLE]
sample = sample.reshape(1, 32, 32, 3)
pred = model.predict(sample)
print("\nPrediction after learning:", labels[np.argmax(pred)], "\n")


# display confusion matrix
# reshaping is necessary to use confusion_matrix function
# we take argmax on one-hot encoding results 
from sklearn.metrics import confusion_matrix
answer = np.argmax(Y_test, axis = 1)

pred = model.predict(X_test)
pred = np.argmax(pred, axis = 1)

print('\n== CONFUSION MATRIX ==')
print(confusion_matrix(answer, pred))


# calculate F1 score using confusion matrix
from sklearn.metrics import f1_score
print("\nF1 score:", f1_score(answer, pred, average = 'micro'))
