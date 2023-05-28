import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

# loads the csv file containing player data into a pandas dataframe
df = pd.read_csv('starcraft_player_data.csv')

# removes all players containing unknown data values, '?', from the dataframe
df = df.where(df != '?').dropna()
df.tail()

# plots a histogram of player density at each LeagueIndex
plt.hist(df["LeagueIndex"])
plt.xlabel('League Index')
plt.ylabel('Number of Players')
plt.title('Number of Players per League Index')
plt.show()

# splits the dataframe into feature variable x_df, last 18 columns, and target variable y_df, LeagueIndex
x_df = df.drop(columns=['GameID', 'LeagueIndex'])
y_df = df['LeagueIndex']

# splits data into two categories, training and testing, 90/10
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.1, random_state=1)

# creates a Regression Tree model to predict the rank of a player, training it with the data from x_train and y_train
model = DecisionTreeRegressor(criterion='friedman_mse', max_depth=10, min_samples_split=10, random_state=1)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print(mean_squared_error(y_test, predictions))
print(np.sqrt(mean_squared_error(y_test, predictions)))

print(cross_val_score(model, x_train, y_train, cv=10))
