# IMDB Movie Review Sentiment Classification with Keras RNN
# July 5, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation, SimpleRNN


# global constants and hyper-parameters
MY_NUM_WORDS = 10000
MY_MAX_LENGTH = 80
MY_DIM = 32
MY_SAMPLE = 2
MY_DROPOUT = 0.2
MY_EPOCH = 5
MY_BATCH = 200


    ####################
    # DATABASE SETTING #
    ####################


# load the DB
# need to downgrade numpy to 1.16.1
# use: pip install numpy == 1.16.1
# "num_words" decides how many most popular words to use in the dictionary
# try 10000 vs 100
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words = MY_NUM_WORDS)


# display DB info
# input is 1-dimensional array of lists
# output is 1-dimensional array of binary numbers
print('\n== DB SHAPE INFO ==')
print('X_train shape = ', X_train.shape)
print('X_test shape = ', X_test.shape)
print('Y_train shape = ', Y_train.shape)
print('Y_test shape = ', Y_test.shape)

print('\n== SAMPLE REVIEW ==')
print(X_train[MY_SAMPLE])
print('Number of words:', len(X_train[MY_SAMPLE]))
print('Sentiment:', Y_train[MY_SAMPLE])


# function to display the length of the first 10 reviews
def show_length():
    print('\n== LENGTH OF THE FIRST 10 REVIEWS ==')
    for i in range(10):
        print("Review", i, "=", len(X_train[i]))

show_length()


# python dictionary: word -> index
# "the", index 1, is the most popular word
# zero index is not used
word_to_id = imdb.get_word_index()
print('\n== DICTIONARY INFO ==')
print("There are", len(word_to_id) + 1, "words in the dictionary.")
print('The index of "hello" is', word_to_id['hello'])


# python dictionary: index -> word
# this is the opposite to word_to_id dictionary
id_to_word = {}
for key, val in word_to_id.items():
    id_to_word[val] = key
print('The word at index 4822 is', id_to_word[4822])


# function to translate the sample review
# we use python dictionary get() function
# it returns "???" if the ID is not found
# index is subtracted by 3 to handle the first 3 special characters
# we use python list and join() function to concatenate the words
def decoding():
    decoded = []

    for i in X_train[MY_SAMPLE]:
        word = id_to_word.get(i - 3, "???")
        decoded.append(word)
        
    print('\n== SAMPLE REVIEW DECODED ==')
    print(" ".join(decoded))

decoding()


# padding to limit the # of words in each review
from keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(X_train, truncating = 'post', padding = 'post', 
        maxlen = MY_MAX_LENGTH)
X_test = pad_sequences(X_test, truncating = 'post', padding = 'post', 
        maxlen = MY_MAX_LENGTH)
decoding()
show_length()



    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# build a keras sequential model of our RNN
# word embedding is key in natural language process (NLP)
# it simplifies vector representation of words
# each word is reduced from MY_NUM_WORDS down to MY_DIM
model = Sequential()
model.add(Embedding(MY_NUM_WORDS, MY_DIM, input_length = MY_MAX_LENGTH))


# size of hidden layer in RNN cell is MY_DIM
# input_shape needs (time_steps, input_dim)
# we use embedded words for the input in this RNN

# dropout: filters input/output synapses
# recurrent dropout: filters synapses between stages
model.add(SimpleRNN(MY_DIM, input_shape = (MY_MAX_LENGTH, MY_DIM),
        dropout = MY_DROPOUT, recurrent_dropout = MY_DROPOUT))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()


# model training and saving
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', 
        metrics = ['accuracy'])
model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = MY_EPOCH, 
        batch_size = MY_BATCH, verbose = 1)
model.save('chap8.h5')


    ####################
    # MODEL EVALUATION #
    ####################


# model evaluation
scores = model.evaluate(X_test, Y_test, verbose = 1)
print("Accuracy: %.2f%%" % (scores[1] * 100))


# display confusion matrix
# the third line converts [0, 1] into true/false
from sklearn.metrics import confusion_matrix
pred = model.predict(X_test, verbose = 1)
pred = (pred > 0.5)
print('\n== CONFUSION MATRIX ==')
print(confusion_matrix(Y_test, pred))


# calculate F1 score using confusion matrix
from sklearn.metrics import f1_score
print("\nF1 score:", f1_score(Y_test, pred, average = 'micro'))
