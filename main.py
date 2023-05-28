import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

# load csv file into a pandas dataframe
df = pd.read_csv('starcraft_player_data.csv')

# remove all players containing unknown data values, '?', from the dataframe
df = df.where(df != '?').dropna()
df.head()

# creating a histogram to map player density per LeagueIndex
plt.hist(df['LeagueIndex'])
plt.xlabel('LeagueIndex')
plt.ylabel('Player Density')
plt.title('Player Density per LeagueIndex')
plt.show()

# the histogram shows no linear trend for LeagueIndex, and we don't know which parameter is
# more or less important to calculating rank, so I decided to use a DecisionTreeClassifier
# model, which will run multiple splits and use the split with the error closest to zero.

# splits the dataframe into feature variable x, last 18 columns, and target variable y, LeagueIndex
x = df.drop(columns=['GameID', 'LeagueIndex'])
y = df['LeagueIndex']

# using a 90/10 split between training data and test data to thoroughly train the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

# creates the DecisionTreeClassifier model and fits it to the training data
model = DecisionTreeClassifier(max_depth=8, random_state=4)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# calculates the accuracy of the model by comparing the predicted rank to the recorded rank
print('Accuracy:', str(round(accuracy_score(y_test, predictions) * 100, 2)) + '%')
