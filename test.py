# YOUR CODE HERE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# print(train.shape)
# print(train.info)
# print(train.head())
# print(train.describe())

num_cols = ['GrLivArea', 'OverallQual', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'LotArea']
cat_cols = ['Neighborhood', 'HouseStyle', 'KitchenQual', 'CentralAir']

features = num_cols + cat_cols
X = train[features].copy()
Y = train['SalePrice'].copy()
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

kitchen_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
X['KitchenQual'] = X['KitchenQual'].map(kitchen_map).fillna(2)

X['CentralAir'] = X['CentralAir'].map({'Y': 1, 'N': 0}).fillna(0)


X = pd.get_dummies(X, columns=['Neighborhood', 'HouseStyle'], drop_first=True)

print(X.isnull())

print(X.shape)

xTrain , xTest , yTrain , yTest = train_test_split(X,Y,test_size=0.20, random_state=42)

model = LinearRegression()
model.fit(xTrain,yTrain)
print(xTrain.shape)
print(xTest.shape)

ypred = model.predict(xTest)
yActual = yTest.iloc[:10].values
print("Actual prices: ",yActual)
print("predicted prices: ",ypred[:10])