import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ames = pd.read_csv("/content/AmesHousing.csv")

ames.shape #81 columns and 2051 records

ames.info() # There are a lot of columns

ames.describe().T #Summary statistics 

ames.isnull().sum()

exc = ames[[ 'Bldg Type', 'Central Air']]

exc







ames = ames.select_dtypes(exclude = ['object'])

ames.columns

len(ames.columns)

# TRAIN TEST SPLIT

ames_copy = ames.copy()

ames_copy = ames_copy.dropna()

ames_copy.isnull().sum()

labels = ames[['SalePrice']]

features = ames.drop(columns=['SalePrice'])

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.25,random_state = 42)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test,y_pred)

mse


# mean_squared_error(wineLabelsTest,quality_Test_Predictions_forest)

# plt.scatter(wineLabelsTest,quality_Test_Predictions_forest)
# plt.plot(wineLabelsTest,wineLabelsTest,"r--")

# Scatter plot

plt.scatter(y_pred,y_test)
plt.plot(y_test,y_test,'r--')

#testing

except_data.dropna()

ames_copy.shape

# ames_copy = ames_copy.join(except_data.set_index(ames_copy.index))

ames_cat=pd.merge(ames_copy, except_data, left_index=True, right_index=True)


ames_cat.isnull().sum()

type(ames_cat)

ames_cat[['Bldg Type']]

# get dummies

ames_oh = pd.get_dummies(ames_cat)

ames_oh
