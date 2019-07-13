# Image Reconstruction with Keras CNN Autoencoder
# July 7, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D
from keras.layers import MaxPooling2D, UpSampling2D


# global constants and hyper-parameters
MY_EPOCH = 10
MY_BATCH = 200
MY_VALID = 0.2


    ####################
    # DATABASE SETTING #
    ####################


# read MNIST hand-written data
# note that we do not need output labels!
# this means we are doing unsupervised learning
(x_train, _), (x_test, _) = datasets.mnist.load_data()


# reshaping 2D image (28 x 28) to 3D image 
# with channel info (28 x 28 x 1)
# we use channel-last ordering (keras default)
# channel = 1 for black/white images
print('\n== SHAPE INFO ==')
print('Train input shape before reshaping:', x_train.shape)
print('Test input shape before reshaping:', x_test.shape)

train_tot = x_train.shape[0]
rows = x_train.shape[1]
cols = x_train.shape[2]
test_tot = x_test.shape[0]

channel = 1
x_train = x_train.reshape(train_tot, rows, cols, channel)
x_test = x_test.reshape(test_tot, rows, cols, channel)

print('Train input shape after reshaping:', x_train.shape)
print('Test input shape after reshaping:', x_test.shape)


# make pixel data float (to compute with float weights)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# we will NOT use keras Sequential model
# note how input ios provided explicitly for each layer
# total parameter count formula =
# (filter_height * filter_width * input_channels + 1) * #_filters


# input layer
shape_in = (28, 28, 1)
image = Input(shape = shape_in)


# 5-layer encoder
# max-pooling reduces image size: 28 -> 14 -> 7
x = Conv2D(10, (3, 3), padding = 'SAME', activation = 'relu')(image)
x = MaxPooling2D((2, 2), padding = 'SAME')(x)
x = Conv2D(10, (3, 3), padding = 'SAME', activation = 'relu')(x)
x = MaxPooling2D((2, 2), padding = 'SAME')(x)
h = Conv2D(1, (7, 7), padding = 'SAME', activation = 'relu')(x)

encoder = Model(image, h)  


# 5-layer decoder
# upscaling increases image size: 7 -> 14 -> 28
y = Conv2D(20, (3, 3), padding = 'SAME', activation = 'relu')(h)
y = UpSampling2D((2, 2))(y)
y = Conv2D(10, (3, 3), padding = 'SAME', activation = 'relu')(y)
y = UpSampling2D((2, 2))(y)
y = Conv2D(5, (3, 3), padding = 'SAME', activation = 'relu')(y)
z = Conv2D(1, (3, 3), padding = 'SAME', activation = 'sigmoid')(y)

model = Model(image, z)
decoder = model
model.summary()


# model training and saving
# conduct training
# note we do not use "y_train", the output label 
# because autoencoder is unsupervised
# Train on 48000 samples, validate on 12000 samples
model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
model.fit(x_train, x_train, epochs = MY_EPOCH, batch_size = MY_BATCH, 
        validation_split = MY_VALID)
model.save('chap5.h5')


    ####################
    # MODEL EVALUATION #
    ####################


# we need to run the autoencoder model using "predict" 
# x_test is the original image
# encoded is the encoder output image
# decoded is the decoder output image
encoded = encoder.predict(x_test)
decoded = decoder.predict(x_test)


# print shape info
print('\n== SHAPE INFO ==')
print('\nOriginal test image shape:', x_test.shape)
print('Encoded test image shape:', encoded.shape)
print('Decoded test image shape:', decoded.shape)


# display function
Nout = 10
plt.figure(figsize = (20, 6))

for i in range(Nout):
    # show the input image
    ax = plt.subplot(3, Nout, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # show the encoded image using histogram
    ax = plt.subplot(3, Nout, i + 1 + Nout)
    plt.imshow(encoded[i].reshape(7, 7))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # show the decoded image
    ax = plt.subplot(3, Nout, i + 1 + Nout + Nout)
    plt.imshow(decoded[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()