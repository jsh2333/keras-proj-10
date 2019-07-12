# Boston Housing Price Prediction with DNN
# June 27, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics


# global constants and hyper-parameters
MY_EPOCH = 500
MY_BATCH = 32


    ####################
    # DATABASE SETTING #
    ####################


# read DB file using pandas
# it retuns a pandas data frame
heading = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 
        'rad', 'tax', 'ptratio', 'black', 'lstat', 'medv']
file_name = "housing.csv"
raw_DB = pd.read_csv(file_name, delim_whitespace = True, names = heading)


# print raw data stats with describe()
print('\n== FIRST 20 RAW DATA ==')
print(raw_DB.head(20))
summary = raw_DB.describe()
summary = summary.transpose()
print('\n== SUMMARY OF RAW DATA ==')
print(summary)


# scaling with z-score: z = (x - u) / s
# mean becomes 0, and standard deviation 1
# it returns numpy array
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_DB = scaler.fit_transform(raw_DB)


# framing numpy array into pandas data frame
# this is needed after scaling to use describe()
scaled_DB = pd.DataFrame(scaled_DB, columns = heading)
summary = scaled_DB.describe()
summary = summary.transpose()
print('\n== SUMMARY OF SCALED DATA ==')
print(summary)


# display box plot of scaled DB
boxplot = scaled_DB.boxplot(column = heading)
print('\n== BOX PLOT OF SCALED DATA ==')
plt.show()


# split the DB into inputs and label
# (506, 14) becomes (506, 13) and (506,)
print('\n== DB SHAPE INFO ==')
print('DB shape = ', scaled_DB.shape)

X = scaled_DB.drop('medv', axis = 1)
print('X (= input) shape = ', X.shape)

Y = scaled_DB['medv']
print('Y (= output) shape = ', Y.shape)


# split the DB into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
        test_size = 0.30, random_state = 5)

print('\nX train shape = ', X_train.shape)
print('X test shape = ', X_test.shape)
print('Y train shape = ', Y_train.shape)
print('Y test shape = ', Y_test.shape)


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# build a keras sequential model of our DNN
model = Sequential()
model.add(Dense(200, input_dim = 13, activation = 'relu'))
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
model.summary()


# model training and saving
model.compile(optimizer = 'adam', loss = 'mean_squared_error', 
        metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = MY_EPOCH, 
        batch_size = MY_BATCH, verbose = 1)
model.save('chap2.h5')


    ####################
    # MODEL EVALUATION #
    ####################


# model evaluation
loss, acc = model.evaluate(X_test, Y_test, verbose = 1)
print('\nMSE of DNN model', loss)
print('Model accuracy', acc)


# comparison with linear regression
from sklearn.linear_model import LinearRegression
model2 = LinearRegression()
model2.fit(X_train, Y_train)
Y_model2 = model2.predict(X_test)
Y_model = model.predict(X_test)


# plot keras DNN modeling result
plt.figure(1)
plt.subplot(121)
plt.scatter(Y_test, Y_model)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Keras DNN Model")


# plot linear regression modeling result
plt.subplot(122)
plt.scatter(Y_test, Y_model2)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Linear Regression Model")
plt.show()


# calculate mean square error of linear regression model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, Y_model2)
print('\nMSE of linear regression model', mse)
