import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load csv file into a pandas dataframe
df = pd.read_csv('starcraft_player_data.csv')

# removes all players containing unknown data values, '?', from the dataframe
df = df.where(df != '?').dropna()
df.head()

# splits the dataframe into feature variable x_df and target variable y_df
x_df = df.drop(columns=['GameID', 'LeagueIndex'])
x_df = x_df.to_numpy().astype(float)

y_df = df['LeagueIndex']
y_df = y_df.to_numpy().astype(float)

# 90/10 split training data and test data
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.1, random_state=1)

# creates a model to predict the rank of the player
model = DecisionTreeClassifier(max_depth=8, random_state=4)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# calculates the accuracy of the model by comparing the prediction to the recorded rank
print('Accuracy:', str(round(accuracy_score(y_test, predictions) * 100, 2)) + '%')
