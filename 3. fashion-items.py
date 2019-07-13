# Fashion MNIST Prediction with CNN
# July 4, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Flatten, Activation
from keras.layers import Dense, MaxPool2D, Conv2D, InputLayer


# global constants and hyper-parameters
TOTAL_CLASS = 10
MY_SAMPLE = 5
MY_EPOCH = 10
MY_BATCH = 200


    ####################
    # DATABASE SETTING #
    ####################


# load DB and split into train vs. test
# fasion MNIST data is 28x28 gray-scale image
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
print('\n== DB SHAPING INFO ==')
print("X train shape:", X_train.shape)
print("Y train shape:", Y_train.shape)
print("X test shape:", X_test.shape)
print("Y test shape:", Y_test.shape)


# define labels
labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle-boot']


# display a sample image and label
print('\n== SAMPLE DATA (RAW) ==')
sample = Y_train[MY_SAMPLE]
print("This is a", labels[sample], sample)
print(X_train[MY_SAMPLE])
plt.imshow(X_train[MY_SAMPLE])
plt.show()


# counting the number of data in each category
unique, counts = np.unique(Y_train, return_counts = True)
print('\n== NUMBER OF DATA IN THE TRAINING SET ==')
for i in range (TOTAL_CLASS):
    print("label", unique[i], ":", counts[i])

unique, counts = np.unique(Y_test, return_counts = True)
print('\n== NUMBER OF DATA IN THE TEST SET ==')
for i in range (TOTAL_CLASS):
    print("label", unique[i], ":", counts[i])


# converting to floats
X_train = X_train / 255.0
X_test = X_test / 255.0


# reshaping before entering CNN
# one-hot encoding is used for the output
# we use channel-last ordering (keras default)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
Y_train = np_utils.to_categorical(Y_train, TOTAL_CLASS)
Y_test = np_utils.to_categorical(Y_test, TOTAL_CLASS)

print('\n== DB SHAPING INFO ==')
print("X train shape:", X_train.shape)
print("Y train shape:", Y_train.shape)
print("X test shape:", X_test.shape)
print("Y test shape:", Y_test.shape)


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# build a keras sequential model of our CNN
# total parameter count formula = 
# (filter_height * filter_width * input_channels + 1) * #_filters
model = Sequential()
model.add(InputLayer(input_shape = (28, 28, 1)))

# no. parameters = (2 x 2 x 1 + 1) x 32
model.add(Conv2D(32, kernel_size = (2, 2), padding = 'same', 
        activation = 'relu'))
model.add(MaxPool2D(padding = 'same', pool_size = (2, 2)))

# no. parameters = (2 x 2 x 32 + 1) x 64
model.add(Conv2D(64, kernel_size = (2, 2), padding = 'same', 
        activation = 'relu'))
model.add(MaxPool2D(padding = 'same', pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))

model.add(Dense(TOTAL_CLASS, activation = 'softmax'))
model.summary()


# testing the model before learning with a sample
sample = X_train[MY_SAMPLE]
sample = sample.reshape(1, 28, 28, 1)
pred = model.predict(sample)
print("\nPrediction before learning:", labels[np.argmax(pred)], "\n")


# model training and saving
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
        metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = MY_EPOCH, batch_size = MY_BATCH, verbose = 1)
model.save('chap3.h5')


    ####################
    # MODEL EVALUATION #
    ####################


# model evaluation
score = model.evaluate(X_test, Y_test, verbose = 1)
print('\nKeras CNN model loss = ', score[0])
print('Keras CNN model accuracy = ', score[1])


# display confusion matrix
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


# testing the model after learning with a sample
sample = X_train[MY_SAMPLE]
sample = sample.reshape(1, 28, 28, 1)
pred = model.predict(sample)
print("\nPrediction after learning:", labels[np.argmax(pred)], "\n")
