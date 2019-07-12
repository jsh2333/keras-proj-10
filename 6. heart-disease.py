# Heart Disease Prediction with DNN
# July 4, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


# global constants and hyper-parameters
INPUT_DIM = 13
MY_EPOCH = 200
MY_BATCH = 32


    ####################
    # DATABASE SETTING #
    ####################


# read DB file
heading = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
        'exang', 'oldpeak', 'slope', 'ca', 'hal', 'HeartDisease']
file_name = "heart.xlsx"
raw_DB= pd.read_excel(file_name, header = None, names = heading)


# print raw data using pandas data frame
# describe() collects DB statistics
print('\n== FIRST 20 RAW DATA ==')
print(raw_DB.head(20))
summary = raw_DB.describe()
print('\n== SUMMARY OF RAW DATA ==')
print(summary)
print('\n== RAW DATA BEFORE CLEAN UP ==')
print(raw_DB.info())


# handling missing entries in the BD
# our DB contains "?". try searching with "~?" in excel.
# first, we replace "?"" with "nan" (not-a-number)
# second, we drop the rows that contain "nan"
clean_DB = raw_DB.replace('?', np.nan)
clean_DB = clean_DB.dropna()
print('\n== RAW DATA AFTER DROPING NAN ROWS ==')
print(clean_DB.info())


# split DB (14 columns) into inputs (13) vs. output (1) first
# so that we scale only the inputs
# output scaling is not useful as it is binary decision
print('\n== DB SHAPE INFO ==')
print('DB shape = ', clean_DB.shape)
keep = heading.pop()
Input = pd.DataFrame(clean_DB.iloc[:, 0:INPUT_DIM], columns = heading)
Target = pd.DataFrame(clean_DB.iloc[:, INPUT_DIM], columns = [keep])

print('\n== INPUT DATABASE AFTER SPLIT ==')
print(Input)
print('\n== OUTPUT DATABASE AFTER SPLIT ==')
print(Target)


# scaling with z-score: z = (x - u) / s
# so that mean becomes 0, and standard deviation 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_DB = scaler.fit_transform(Input)


# collect scaled DB stats using described()
# framing is needed after scaling
scaled_DB = pd.DataFrame(scaled_DB, columns = heading)
summary = scaled_DB.describe()
summary = summary.transpose()
print('\n== SUMMARY OF SCALED DATA ==')
print(summary)


# display box plot of scaled DB
boxplot = scaled_DB.boxplot(column = heading, showmeans = True)
print('\n== BOX PLOT OF SCALED DATA ==')
plt.show()


# split the DB into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(scaled_DB, Target, 
        test_size = 0.30, random_state = 5)
print('X train shape = ', X_train.shape)
print('X test shape = ', X_test.shape)
print('Y train shape = ', Y_train.shape)
print('Y test shape = ', Y_test.shape)


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# build a keras sequential model of our DNN
model = Sequential()
model.add(Dense(1000, input_dim = INPUT_DIM, activation = 'tanh'))
model.add(Dense(1000, activation = 'tanh'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()


# model training and saving
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
        metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = MY_EPOCH, batch_size = MY_BATCH, 
        verbose = 1)
model.save('chap6.h5')


    ####################
    # MODEL EVALUATION #
    ####################


# model evaluation
score = model.evaluate(X_test, Y_test, verbose = 1)
print('\nKeras DNN model loss = ', score[0])
print('Keras DNN model accuracy = ', score[1])


# display confusion matrix
# the third line converts [0, 1] into true/false
from sklearn.metrics import confusion_matrix
pred = model.predict(X_test)
pred = (pred > 0.5)
print('\n== CONFUSION MATRIX ==')
print(confusion_matrix(Y_test, pred))


# calculate F1 score using confusion matrix
from sklearn.metrics import f1_score
print("\nF1 score:", f1_score(Y_test, pred, average = 'micro'))

