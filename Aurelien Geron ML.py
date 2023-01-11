import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

data = pd.read_csv(data_url,sep=';')

data.head()

### Features

feature_list = data.columns[:-1].values

label = [data.columns[-1]]

feature_list

print(type(feature_list))

### data statistics

data.info()

# Total entries : 1599 (Tiny Dataset by ML Standard)
# There are total 12 columns : 11 Features + 1 Label

#   -Features :   'fixed acidity', 'volatile acidity', 'citric acid',
#                'residual sugar', 'chlorides', 'free sulfur dioxide',
#                'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
               
#   -Label    :   quality

# # to understand decriptive statistics - mean sd quartiles
# data.describe()

# quality :

# min : 3

# 25% : 5

# 50% : 6

# 75% : 6

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

# We can create subplots in Python using matplotlib with the subplot method, which takes three arguments:

# nrows: The number of rows of subplots in the plot grid.
# ncols: The number of columns of subplots in the plot grid.
# index: The plot that you have currently selected.

data.columns

plt.subplot(3,3,1)
plt.hist(data['total sulfur dioxide'])
plt.title("total sulfur dioxide")


plt.subplot(3,3,2)
plt.hist(data.density)
plt.title("density ")

plt.subplot(3,3,3)
plt.hist(data.pH)
plt.title(" pH")


plt.subplot(3,3,4)
plt.hist(data.sulphates)
plt.title("sulphates ")


plt.subplot(3,3,5)
plt.hist(data.alcohol)
plt.title("alcohol ")


plt.subplot(3,3,6)
plt.hist(data.quality)
plt.title(" quality")
plt.gcf().set_size_inches(20, 20)

# GCF -  Get the current figure.

data.hist(bins=50, figsize=(20,15))
plt.show()

If the test data is used in building the model then that is known as "data snooping"

If the test data has affected any step in learning process,the model's ability to access the outcome has been compromised that is referred  as "data snooping"


1 . Data visulaization  : lower dimensional data includes testing data also
2 . Data Pre Processing : 

np.random.seed(0)
np.random.randint(10, size = 5)

np.random.seed(0)
np.random.randint(10, size = 5)
# The numpy.random.seed function provides the input (i.e., the seed) to the algorithm that generates pseudo-random numbers in NumPy.
# NUMPY RANDOM SEED MAKES YOUR CODE REPEATABLE


def split_train_test(data,testRatio):
    np.random.seed(37)
    
    shuffledIndices = np.random.permutation(len(data))
    
    test_set_size = int(len(data)*testRatio)
    
    testIndices = shuffledIndices[:test_set_size]
    trainIndices = shuffledIndices[test_set_size:]
    return data.iloc[trainIndices], data.iloc[testIndices]

split_train_test(data,0.2)

train_set,test_set = split_train_test(data,0.2)

train_set

test_set

Sci-kit learn provides :
	1. Random Sampling : which randomly selects k% points in the test set
	2. Stratifies Sampling : which samples test examples such that they are representative of overall distribution

	
	
 train_test_split ( )  function performs random sampling with 

	○ random_state : parameter to set random seed , which ensures that the same examples are selected for test set across runs
	○  test_size : parameter for specifying size of the test set
	○  shuffle flag : to specify whether the data needs to be split before splitting


from sklearn.model_selection import train_test_split

?train_test_split

train_set,test_set = train_test_split(data,test_size=0.2,random_state=42)

from sklearn.model_selection import StratifiedShuffleSplit 

sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in sss.split(data,data['quality']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

n_splits - number of splits (here we need 1 split as train and test)

sss.split(paramater 1,parameter 2 ) 
so that based on 2nd parameter the first parameter will be splitted
so that data will be splitted based on data['quality']


sss.split will return 2 lists 
1st list contain the indices of train set
2nd list contain the indices of test set

loc returns the rows with the specified index
and we are saving it in strat_train_set and strat_test_set

loc is label-based, which means that you have to specify rows and columns based on their row and column labels.

iloc is integer position-based, so you have to specify rows and columns by their integer position values (0-based integer position)

strat_dist = strat_test_set['quality'].value_counts()/len(strat_test_set)

strat_dist

random_dist = test_set['quality'].value_counts()/len(test_set)

random_dist

overall_dist = data['quality'].value_counts()/len(data)

distComparison = pd.DataFrame(
    {"Overall ":overall_dist,
     
     "Random" : random_dist,
    "Stratified":strat_dist})

distComparison['diff[S-O]Error'] = distComparison['Stratified']-distComparison['Overall ']
distComparison['diff[S-O]Error %'] = 100*(distComparison['diff[S-O]Error'] / distComparison['Stratified'])

distComparison['diff[R-O]Error'] = distComparison['Random']-distComparison['Overall ']
distComparison['diff[R-O]Error %'] = 100*(distComparison['diff[R-O]Error'] / distComparison['Random'])

distComparison

This shows the proportions in the overall dataset,
in the test set generated with stratified sampling, and in a test set generated using
purely random sampling. As you can see, the test set generated using stratified
sampling has income category proportions almost identical to those in the full
dataset, whereas the test set generated using purely random sampling is quite
skewed.

distComparison.loc[:,['diff[S-O]Error %','diff[R-O]Error %']]

# Lecture 2
Data viz

explorationSet = strat_train_set.copy()

Scatter plot

sns.scatterplot(x='fixed acidity',y='density',hue='quality',data=explorationSet)
plt.gcf().set_size_inches(10,10)

Matplotlib

explorationSet.plot(kind='scatter',x='fixed acidity',y='density',c='quality',alpha=0.1,colormap='twilight')
# Setting the alpha option to 0.1 makes it much easier to
# visualize the places where there is a high density of data points

hsv 
twilight
tab20c
flag
rainbow
turbo

https://matplotlib.org/stable/tutorials/colors/colormaps.html



Relationship between features

1 . Standard Correlation Coefficient  - ranges between -1 to +1
correlation +1 : Strong positive correlation between features
correlation -1 : Strong negative correlation between features
correlation  0 : No linear correlation between features


Viz correlation via heatmap

for non - linear relationship use rank correlation

corr_matrix = explorationSet.corr()

corr_matrix

corr_matrix['quality']

corr_matrix['quality'].sort_values(ascending=False)

The correlation coefficient only measures linear correlations
(“if x goes up, then y generally goes up/down”

so i can say that alcohol has something to say about the quality of wine 
whereas the volatile acidity has nothing to say  about the quality of wine 

using heatmap to viz correlation

Heatmap is a symmetric matrix

sns.heatmap(corr_matrix,annot=True)
plt.gcf().set_size_inches(20,20)
# plt.figure(figsize=(50,50))

Another way to check for correlation between attributes is to use Pandas’
scatter_matrix function, which plots every numerical attribute against every
other numerical attribute

Since there are now 11 numerical attributes, you would
get 112 = 121 plots, which would not fit on a page, so let’s just focus on a few
promising attributes that seem most correlated

from pandas.plotting import scatter_matrix

attribute_list = ['citric acid','pH',"alcohol","sulphates","quality"]
scatter_matrix(explorationSet[attribute_list])
plt.gcf().set_size_inches(10,10)

Similar analysis can be carried out using combined features - features that are derived from the original features

The main diagonal (top left to bottom right) would be full of straight lines if
Pandas plotted each variable against itself, which would not be very useful. So
instead Pandas displays a histogram of each attribute

One last thing you may want to do before actually preparing the data for Machine Learning algorithms is to try out various attribute combinations. For example, the total number of rooms in a district is not very useful if you don’t
know how many households there are. What you really want is the number of rooms per household. Similarly, the total number of bedrooms by itself is not very useful: you probably want to compare it to the number of rooms. And the
population per household also seems like an interesting attribute combination to look at.


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]




The new bedrooms_per_room attribute is much more correlated

Lecture 3

But first let’s revert to a clean training set (by copying strat_train_set once again) and 
let’s separate the predictors and the labels

drop() creates a copy of the data and does not affect strat_train_set:

wineFeatures = strat_train_set.drop("quality", axis=1)
wineLabels = strat_train_set['quality'].copy()

wineFeatures

wineLabels

# data cleaning

# Counts the number of NaN in each column of wineFeatures
wineFeatures.isna().sum()

	In case of missing  values :
	• Use imputation method to fill up the missing values
	• Drop the rows containing missing values

Sklearn provides following methods to drop  rows containing missing values:
	•  dropna()
	•  drop()
	• It provides SimpleImputer class for filling up the missing values (say median values)


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')

imputer.fit(wineFeatures)

In Case , the feature contains non-numeric attributes , they need to be dropped before calling the fit method on imputer object

imputer.statistics_

wineFeatures.columns

wineFeatures.median()

wineFeatures.median()

Finally we use imputer to transform the training set such that the missing values are replaced by median

tr_features = imputer.transform(wineFeatures)

tr_features.shape

print(type(tr_features))

This returns a Numpy array and we can convert it to dataframe if needed

wineFeatures_tr = pd.DataFrame(tr_features,columns=wineFeatures.columns)

wineFeatures_tr



**Text - NumberClasses - One hot encoded vector**

**Text - NumberClasses:**
*OrdinalEncoder and 
LabelEncoder*


**NumberClasses - One hot encoded vector**
*OneHotEncoder*

**Text - One hot encoded vector** : *LabelBinarizer*

from sklearn.preprocessing import OrdinalEncoder
ordinalEncoder = OrdinalEncoder()

call fit_transform() method on OrdinalEncoder object to convert text to numbers

The list of categories can be obtained via  categories_instance variable

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

You can look at the mapping that this encoder has learned using the classes_ attribute
print(encoder.classes_)

One issue with this representation is that ML algorithms will assume that two
nearby values are more similar than two distant values. Obviously this is not the
case (for example, categories 0 and 4 are more similar than categories 0 and 1).
To fix this issue, a common solution is to create one binary attribute per
category: one attribute equal to 1 when the category is “<1H OCEAN” (and 0
otherwise), another attribute equal to 1 when the category is “INLAND” (and 0
otherwise), and so on. This is called one-hot encoding

create one binary attribute per category:
Onlu one feature is 1 (hot) and the rest are 0 (cold)

The new features are referred as Dummy Features

Sci-kit learn provides a OneHotEncoder class to convert categorical values into one-hot vectors

from sklearn.preprocessing import OneHotEncoder
hotEncoder = OneHotEncoder()

wine_1hot = hotEncoder.fit_transform(wineFeatures)

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

wine_1hot.toarray()

Notice that the output is a SciPy sparse matrix, instead of a NumPy array. This is
very useful when you have categorical attributes with thousands of categories.
After one-hot encoding we get a matrix with thousands of columns, and the
matrix is full of zeros except for one 1 per row. Using up tons of memory mostly
to store zeros would be very wasteful

so instead a sparse matrix only stores the  ***location*** of the nonzero elements.

call fit_transform() method on OneHotEncoder object 
Features with sparse data are features that have mostly zero values
The otput is a SciPy sparse matrix rather than NumPy array
This enables us to save space when we have a huge number of categories


NumPy Arrays are multi-dimensional arrays of objects which are of the same type i.e.  homogeneous.

SciPy does not have any such array concepts as it is more functional. It has no constraints of homogeneity.

In case we want to convert it to dense representation , we can do so using toarray() method


The list of categories can be obtained via  categories_



Scikit-Learn provides a OneHotEncoder encoder to convert integer categorical values into one-hot vectors



When the number of categories are very large, the one-hot encoding would result in a very large number of features

This could be addressed with one of the following approaches :

1 . Replace with categorical numerical features

2 . Convert into *low-dimensional* learnable vectors called *embeddings*

We can apply both transformations (from text categories to integer categories,
then from integer categories to one-hot vectors) in one shot using the
LabelBinarizer class


from sklearn.preprocessing import LabelBinarizer
labelEncoder = LabelBinarizer()


Note that this returns a dense NumPy array by default. You can get a sparse
matrix instead by passing sparse_output=True to the LabelBinarizer
constructor.

**Feature Scaling**
	• Machine Learning algorithms don’t perform well when the input numerical attributes have very different scales.
	• Note that scaling the target values is generally not required.
	
	There are two common ways to get all attributes to have the same scale:
	 minmax scaling (Normalization) and StandardScaler (Standardization)
	
	MinMaxScaler scaling:
	• subtracting the min value  from the current value and dividing by the max minus the min.
	• Scikit-Learn provides a transformer called  MinMaxScaler .
	• It has a feature_range hyperparameter that lets you change the range if you don’t want 0–1 


	StandardScaler 

	○ Standardization is quite different: first it subtracts the mean value (so standardized values always have a zero mean), and then it divides by the ariance so that the resulting distribution has unit variance. 
	○ Unlike min-max scaling, standardization does not bound values to a specific range, which may be a problem for some algorithms 
    e.g., neural networks often expect an input value ranging from 0 to 1).
	○  However, standardization is much less affected by outliers.
	Scikit-Learn provides a transformer called StandardScaler for standardization.
	
It is important to fit the scalers to the training data only, not to the full dataset (including the test set). Only then can you use them to
transform the training set and the test set (and new data).



Pipeline has sequence of transformations - missing values imputation followed by Strandardization

**Transformation Pipelines**

As you can see, there are many data transformation steps that need to be **executed in the right order.**
 Fortunately, Scikit-Learn provides the Pipeline class to help with such sequences of transformations.

The Pipeline constructor takes a list of **name/estimator** pairs defining a sequence of steps.
All but the last estimator must be transformers (i.e., they must have a fit_transform() method).
	
When you call the pipeline’s fit() method, it calls fit_transform()
sequentially on all transformers passing the output of each call as the parameter
to the next call, until it reaches the final estimator

Mixed features :
ColumnTransformer
    
    Scikit-Learn provides a FeatureUnion class  You now have a 
    ○ pipeline for numerical values, and 
    ○ LabelBinarizer on the categorical values

	from sklearn.pipeline import FeatureUnion
	
	num_attribs = list(housing_num)
	
	cat_attribs = ["ocean_proximity"]
	
	num_pipeline = Pipeline([
				('selector', DataFrameSelector(num_attribs)),
				('imputer', Imputer(strategy="median")),
				('attribs_adder', CombinedAttributesAdder()),
				('std_scaler', StandardScaler()),
	])
	
	
	cat_pipeline = Pipeline([
				('selector', DataFrameSelector(cat_attribs)),
				('label_binarizer', LabelBinarizer()),
	])
	
	
	full_pipeline = FeatureUnion(transformer_list=[
				("num_pipeline", num_pipeline),
				("cat_pipeline", cat_pipeline),
	])
	
	housing_prepared = full_pipeline.fit_transform(housing)
	housing_prepared
	
	
	
	


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

transformation_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ("scaler",StandardScaler())
])

Mixed features : ColumnTransformer

num_attributes = list(wineFeatures)
cat_attributes = ['place_of_manufacturing'] # say
fullPipeline = ColumnTransformer([
    ('num',num_pipeline,num_attributes),
    ("cat",cat_pipeline,cat_attributes)
])

wine_features_tr = fullPipeline.fit_transform(wineFeatures)

**5.Select and Train a Model**

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(wineFeatures_tr,wineLabels)

# evaluate performance 
# MSE

from sklearn.metrics import mean_squared_error
qualityPredictions = lin_reg.predict(wineFeatures_tr)

mse = mean_squared_error(wineLabels,qualityPredictions)

mse

# apply transformation upon test set and apply model prediction

strat_test_set

wineFeaturesTest = strat_test_set.drop("quality",axis=1)
wineLabelsTest = strat_test_set['quality'].copy()



wineFeatures_Test_tr = transformation_pipeline.fit_transform(wineFeaturesTest)

quality_Test_Predictions = lin_reg.predict(wineFeatures_Test_tr)

mseTest = mean_squared_error(wineLabelsTest,quality_Test_Predictions)

mseTest

mse

plt.scatter(wineLabelsTest,quality_Test_Predictions)
plt.plot(wineLabelsTest,wineLabelsTest,"r--")
plt.xlabel("Actual Quality")
plt.ylabel('Predicted quality')

lets make another model  Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(wineFeatures_tr,wineLabels)

This is a powerful model, capable of  finding complex nonlinear relationships in the data

qualityPredictions_tree = tree_reg.predict(wineFeatures_tr)

mean_squared_error(wineLabels,qualityPredictions_tree)

overfitted model

plt.scatter(wineLabels,qualityPredictions_tree)
plt.plot(wineLabelsTest,wineLabelsTest,"r--")

cross validation for robust evaluation

One way to evaluate the Decision Tree model would be to use the
train_test_split function to split the training set into a smaller training set
and a validation set, then train your models against the smaller training set and
evaluate them against the validation set. It’s a bit of work, but nothing too
difficult and it would work fairly well.
A great alternative is to use Scikit-Learn’s cross-validation feature. The
following code performs K-fold cross-validation: it randomly splits the training
set into 10 distinct subsets called folds, then it trains and evaluates the Decision
Tree model 10 times, picking a different fold for evaluation every time and
training on the other 9 folds.

Scikit-Learn cross-validation features expect a utility function (greater is better)
rather than a cost function (lower is better), so the scoring function is actually the
opposite of the MSE (i.e., a negative value), which is why the preceding code
computes -scores before calculating the square root

from sklearn.model_selection import cross_val_score

CV provides a separate MSE for each validation set, which we can use to get a mean estimation of MSE as well as the SD 

The additional costs that we pay in CV is additional training runs, which may be too expensive in certain cases

def display_scores(scores):
    print("scores : ",scores)
    print("mean : ",scores.mean())
    print("SD : ",scores.std())

scores_lr = cross_val_score(lin_reg,wineFeatures_tr,wineLabels,scoring='neg_mean_squared_error',cv=10)

lin_reg_mse_scores = -scores_lr

display_scores(lin_reg_mse_scores)

scores_dtree = cross_val_score(tree_reg,wineFeatures_tr,wineLabels,scoring='neg_mean_squared_error',cv=10)

tree_mse_scores = -scores_dtree

display_scores(tree_mse_scores)

Lin.Reg has better MSE compared to DTree

Random Forests work by training many Decision Trees on random  subsets of the features, then averaging out their predictions

Building a model on  top of many other models is called Ensemble Learning, and it is often a great
way to push ML algorithms even further.

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(wineFeatures_tr,wineLabels)

scores_forest = cross_val_score(forest_reg,wineFeatures_tr,wineLabels,scoring='neg_mean_squared_error',cv=10)

forest_mse = - scores_forest

display_scores(forest_mse)

quality_Test_Predictions_forest = forest_reg.predict(wineFeatures_Test_tr)

mean_squared_error(wineLabelsTest,quality_Test_Predictions_forest)

plt.scatter(wineLabelsTest,quality_Test_Predictions_forest)
plt.plot(wineLabelsTest,wineLabelsTest,"r--")

import joblib

joblib.dump(lin_reg, "wineQuality_LR.pkl")

my_model_loaded = joblib.load("wineQuality_LR.pkl")

my_model_loaded

# Fine Tuning the model

Usually there are a number of hyperparameters in the model,which are set manually

Tuning this hyperparameters lead to better accuracy of ML Model

Finding the best combination of HYPERPARAMETERS is a search problem in the space of hyperparameters, which is huge

SKLearn provides a class GridSearchCV 

we need to specify a list of hyperparameters along with the range of values to try

It automatically evaluates all possible combinations of hyperparameter values using cross-validation

from sklearn.model_selection import GridSearchCV

param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

#  Total Combination 
 
#  the first one has 3 X 4 = 12 combinations
#  the second one has 2 X 3 = 6 combinations
 
#  so 12 + 6 = 18
 

grid_search_linreg = GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)

# cv = 5 and hp = 18
# so Total model training Runs = 18 X 5 = 90

grid_search_linreg.fit(wineFeatures_tr,wineLabels)

grid_search_linreg.best_params_

cv_res = grid_search_linreg.cv_results_

cv_res

for mean_score,params in zip(cv_res['mean_test_score'],cv_res['params']):
    print(-mean_score,params)

# 0.356880841503268 {'max_features': 8, 'n_estimators': 30}
# is the value that has LOWEST MSE

grid_search_linreg.best_estimator_

# Randomized SearchCV 

# The grid search approach is fine when you are exploring relatively few
# combinations, like in the previous example, but when the hyperparameter search
# space is large, it is often preferable to use RandomizedSearchCV instead.

#
# This class can be used in much the same way as the GridSearchCV class, but instead
# of trying out all possible combinations, it evaluates a given number of random
# combinations by selecting a random value for each hyperparameter at every
# iteration. This approach has two main benefits:


# If you let the randomized search run for, say, 1,000 iterations, this approach
# will explore 1,000 different values for each hyperparameter (instead of just
# a few values per hyperparameter with the grid search approach).


# You have more control over the computing budget you want to allocate to
# hyperparameter search, simply by setting the number of iterations

from sklearn.model_selection import RandomizedSearchCV

# Ensemble Methods


# Another way to fine-tune your system is to try to combine the models that
# perform best. The group (or “ensemble”) will often perform better than the best
# individual model (just like Random Forests perform better than the individual
# Decision Trees they rely on), especially if the individual models make very
# different types of errors

# Analyze the Best Models and Their Errors

feature_importances = grid_search_linreg.best_estimator_.feature_importances_

feature_importances

# member varible

#Let’s display these importance scores next to their corresponding attribute names:

sorted(zip(feature_importances,feature_list),reverse=True)

# With this information, you may want to try dropping some of the less useful features

# It is also useful to analayze the errors in prediction and understand its causes and fix them

# EVALUATION ON TEST SET

wine_features_test = strat_test_set.drop("quality",axis=1)

wine_labels_test = strat_test_set['quality'].copy()

wine_features_test_tr = transformation_pipeline.fit_transform(wine_features_test)

quality_Test_Predictions_eval = grid_search_linreg.best_estimator_.predict(wine_features_test_tr)

mean_squared_error(wine_labels_test,quality_Test_Predictions_eval)

# confidence interval

# from scipy import stats
# confidence = 0.95
# squared_errors = (quality_Test_Predictions_eval - wine_features_test)**2
# stats.t.interval(confidence,len(squared_errors)-1,
#                 loc=squared_errors.mean(),
#                 scale=stats.sem(squared_errors))



