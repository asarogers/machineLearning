import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv("dataset/500_Person_Gender_Height_Weight_Index.csv")
datasetTest = pd.read_csv("dataset/SOCR-HeightWeight.csv")

model = linear_model.LinearRegression()
lin_reg = LinearRegression()

dataset["Gender"].replace("Female", 0,  inplace = True)
dataset['Gender'].replace('Male',1, inplace=True)
dataset = dataset.assign(Height = lambda x: x["Height"]*0.0328084)
dataset = dataset.assign(Weight = lambda x: x["Weight"]*2.2)
dataset  = dataset.loc[dataset["Index"]  <= 4 ]
# dataset = dataset.assign(Weight = lambda x: x["Weight"]/0.0328084)

x = dataset.iloc[:, : 2].values # weight & gender
y = dataset.iloc[:, 2].values #height

# # # train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 5)

lin_reg.fit(x_train, y_train)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
lin_pred = lin_reg.predict(x_test)

weight_pred = model.predict([[1,5.83333333]])
my_weight_pred = lin_reg.predict([[1,74/12]])
female_pred = lin_reg.predict([[0,61/12]])


print("6'2, male = ", my_weight_pred[0])
print("5'10, male = ", weight_pred[0])
print("5'1, female = ", female_pred[0])