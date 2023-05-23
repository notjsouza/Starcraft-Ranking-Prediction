import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tensorflow import keras

# load csv file into a pandas dataframe
data = pd.read_csv('starcraft_player_data.csv')

# removes all players containing unknown values, '?', from the dataframe
data = data.where(data != '?').dropna()
data.head()

encoder = preprocessing.LabelEncoder()
data['LeagueIndex'] = encoder.fit_transform(data['LeagueIndex'])

# convert pandas dataframe to numpy vector
np_stats = data.to_numpy().astype(float)
#
#
#
# extract feature variable x
x_data = np_stats[:, 1:19]

# extract target variable y
y_data = np_stats[:, 1]

#  --------------------- FIX Y_DATA PROCESSING ISSUE!!!!! ----------------------------------------

# convert to one-hot-encoding
y_data = tf.keras.utils.to_categorical(y_data, 1)
print(y_data)

# split training and testing data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

# ------------------------------------------------------------------------------------------------

# training parameters
EPOCHS = 20
BATCH_SIZE = 64
VERBOSE = 1
OUTPUT_CLASSES = len(encoder.classes_)
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

# creating a sequential model
model = tf.keras.models.Sequential()

# creating dense layers
model.add(keras.layers.Dense(N_HIDDEN, input_shape=(18,), name='Dense-Layer-1', activation='relu'))
model.add(keras.layers.Dense(N_HIDDEN, name='Dense-Layer-2', activation='relu'))
model.add(keras.layers.Dense(N_HIDDEN, name='Dense-Layer-3', activation='relu'))
model.add(keras.layers.Dense(OUTPUT_CLASSES, name='Dense-Layer-Final', activation='softmax'))

# compiling the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------- RESTORE WHEN DATA PROCESSING ISSUE IS FIXED -------------------------

# model.summary()

# building the model
# model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# testing the model
# print(model.evaluate(x_test, y_test))

# ------------------------------------------------------------------------------------------------
