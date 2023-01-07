# data pre processing

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

data = pd.read_csv(data_url, sep=';')

data.head()

### Features

feature_list = data.columns[:-1].values

label = [data.columns[-1]]

feature_list

print(type(feature_list))

### data statistics

data.info()

# Total entries: 1599(Tiny  Dataset by ML Standard)  There  are total 12  columns: 11 Features + 1 Label
#
# -Features: 'fixed acidity', 'volatile acidity', 'citric acid',
# 'residual sugar', 'chlorides', 'free sulfur dioxide',
# 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
#
# -Label: quality


# to understand decriptive statistics - mean sd quartiles
data.describe()

# quality :
#
# min : 3
#
# 25% : 5
#
# 50% : 6
#
# 75% : 6
#
# max : 8

## distribution of the wine quality
## datatype : <class 'pandas.core.series.Series'>

# data['quality']
data['quality'].value_counts()

# A Histogram gives the count of how many samples occurs within a specific range(Bins)

sns.set()
data['quality'].hist()
plt.xlabel("Wine Quality")
plt.ylabel("Frequency")

data['fixed acidity'].hist()

# We can create subplots in Python using  matplotlib with the subplot method, which takes three arguments:

# We can create subplots in Python using matplotlib with the subplot method, which takes three arguments:
#
# nrows: The number of rows of subplots in the plot grid.
# ncols: The number of columns of subplots in the plot grid.
# index: The plot that you have currently selected.

data.columns

plt.subplot(3, 3, 1)
plt.hist(data['total sulfur dioxide'])
plt.title("total sulfur dioxide")

plt.subplot(3, 3, 2)
plt.hist(data.density)
plt.title("density ")

plt.subplot(3, 3, 3)
plt.hist(data.pH)
plt.title(" pH")

plt.subplot(3, 3, 4)
plt.hist(data.sulphates)
plt.title("sulphates ")

plt.subplot(3, 3, 5)
plt.hist(data.alcohol)
plt.title("alcohol ")

plt.subplot(3, 3, 6)
plt.hist(data.quality)
plt.title(" quality")
plt.gcf().set_size_inches(20, 20)

# GCF -  Get the current figure.

data.hist(bins=50, figsize=(20, 15))
plt.show()

# If the test data is used in building the model then that is known as "data snooping"
#
# If the test data has affected any step in learning process,the model's ability to access the outcome has been compromised that is referred  as "data snooping"
#
#
# 1 . Data visulaization  : lower dimensional data includes testing data also
# 2 . Data Pre Processing : processing the data into insights

