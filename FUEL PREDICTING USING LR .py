import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns

import numpy as np
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("fueldata.csv")


df

plt.plot(df.drivenKm,df.fuelAmount,"ro")


sns.relplot(x=df.drivenKm,y=df.fuelAmount)


x_df = df[['drivenKm']]
y_df = df[['fuelAmount']]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_df,y_df,test_size=0.20)

print(x_train,type(x_train))
print(y_train,type(y_train))

print(x_test,type(x_test))
print(y_test,type(y_test))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

The unified ML Interface of Sci-kit learn

1 . create  algorithm
2 . call fit function
3 . call predict function

lreg = LinearRegression()

lreg.fit(x_train,y_train)

pred = lreg.predict([[800]])

pred

y_pred = lreg.predict(x_test)

y_pred

y_test

lreg.coef_

lreg.intercept_

mse = mean_squared_error(y_test,y_pred)

mse

y_pred_df = lreg.predict(x_df)
y_pred_df

y_df

plt.plot(df.drivenKm,df.fuelAmount,"r*--")
plt.plot(df.drivenKm,y_pred_df,"g*--")

lreg.score(x_test,y_test)

Normalization is used when we want to bound our values between two numbers, typically, between [0,1] or [-1,1].

Standardization transforms the data to have zero mean and a variance of 1, they make our data unitless.

Feature scaling is essential for machine learning algorithms that calculate distances between data.
If not scale, the feature with a higher value range starts dominating when calculating distances

https://medium.com/towards-data-science/all-about-feature-scaling-bcc0ad75cb35

Why we use fit_transform() on training data but transform() on the test data?

We all know that we call fit_transform() method on our training data and transform()




Standard scaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

norm_x_train = scaler.fit_transform(x_train)
norm_y_train = scaler.fit_transform(y_train)
norm_x_test = scaler.transform(x_test)
norm_y_test = scaler.transform(y_test)

norm_y_train

norm_y_test = scaler.transform(y_test)

norm_lreg = LinearRegression()

norm_lreg.fit(norm_x_train,norm_y_train)

norm_y_pred = norm_lreg.predict(norm_x_test)

norm_mse = mean_squared_error(norm_y_test,norm_y_pred)

norm_mse

plt.plot(norm_y_test,norm_y_pred,"go")



min max scalar

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()

mm_norm_x_train = minmax.fit_transform(x_train)
mm_norm_y_train = minmax.fit_transform(y_train)
mm_norm_x_test = minmax.transform(x_test)
mm_norm_y_test = minmax.transform(y_test)

mm_norm_y_train

mm_norm_x_train

mm_norm_y_test = minmax.transform(y_test)

mm_norm_lreg = LinearRegression()

mm_norm_lreg.fit(mm_norm_x_train,mm_norm_y_train)

mm_norm_y_pred = mm_norm_lreg.predict(mm_norm_x_test)

mm_norm_y_pred

mm_norm_y_test

minmax_norm_mse = mean_squared_error(mm_norm_y_test,mm_norm_y_pred)

minmax_norm_mse

mse


#### prepare the model with input scaling
pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', LinearRegression())])

#### fit pipeline
pipeline.fit(train_x, train_y)

#### make predictions
yhat = pipeline.predict(test_x)

y_test = y_test.to_numpy()



plt.xlabel("y_test")
plt.xlabel("y_pred")
plt.plot(y_test,y_pred,"ro")


plt.plot(norm_y_test,norm_y_pred,"go")

knn regressor

from sklearn.neighbors import KNeighborsRegressor

# creating Instance for the model

knn = KNeighborsRegressor(n_neighbors=5)

# Training / Fitting Data

knn.fit(x_train,y_train)


print(knn.predict([[800]]))

knn_y_pred = knn.predict(x_test)

knn_mse = mean_squared_error(knn_y_pred,y_test)

knn_mse

from sklearn.pipeline import make_pipeline

max_iter = np.ceil(10**6/x_train.shape[0])

sgd = make_pipeline(StandardScaler(),linear_model.SGDRegressor(max_iter = max_iter,tol=1e-3))

print(type(x_train))

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()


sgd.fit(x_train,y_train)

sgd_y_pred = sgd.predict(x_test)

sgd_y_pred

sgd_mse = mean_squared_error(y_test,sgd_y_pred)

sgd_mse

from tabulate import tabulate

data = [["MODELS","MSE VALUE"],
       ["LINEAR REGRESSION",round(mse)],
       ["STANDARD SCALER LR ",round(norm_mse)],
       [" MINMAX  LR",round(minmax_norm_mse) ],
       ["KNN",round(knn_mse)],
       ["SGD",round(sgd_mse)]]

print(tabulate(data))

score_lr = lreg.score(x_test,y_test)
score_norm_lreg = norm_lreg.score(x_test,y_test)
score_mm_norm_lreg = mm_norm_lreg.score(x_test,y_test)
score_knn = knn.score(x_test,y_test)
score_sgd = sgd.score(x_test,y_test)

scores = [['MODELS','SCORES'],
         ["LINEAR REGRESSION",score_lr],
       ["STANDARD SCALER LR ",score_norm_lreg],
       [" MINMAX  LR",score_mm_norm_lreg ],
       ["KNN",score_knn],
       ["SGD",score_sgd]]

print(tabulate(scores))

SGD IS BETTER MODEL WITH LOWEST MSE AND HIGHEST SCORE





