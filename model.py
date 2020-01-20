import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('train.csv')

#Data Cleaning
df = df.drop(['Cabin'], axis=1)
df = df.drop(['Ticket'], axis=1)
df = df.drop(['Name'], axis=1)
df = df.fillna({"Embarked": "S"})

#Replace NAN age values with average age
df['Age'].fillna((df['Age'].mean()), inplace=True)

sex_mapping = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(sex_mapping)

embarked_mapping = {"S": 1, "C": 2, "Q": 3}
df['Embarked'] = df['Embarked'].map(embarked_mapping)

#Splitting data
from sklearn.model_selection import train_test_split
X = df.drop(['Survived', 'PassengerId', 'Parch'], axis=1)
y = df['Survived']
#X = X_old.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

#Building Model
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print("Model Score: ", score)

pickle.dump(gbc, open('model.pkl', 'wb'), protocol=2)

model = pickle.load(open('model.pkl', 'rb'))
#print(model.predict([[1,1,1,1,1,1,1]]))
