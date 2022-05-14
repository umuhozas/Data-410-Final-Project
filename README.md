#### Data-410-Final-Project
#### Solange Umuhoza
#### 11 May 2022

# How Happy Are We Actually?

### Introduction

The goal of this project is to explain and predict happiness around the world. This project will use a "World Happiness Report up to 2022" dataset found on Kaggle.com and it contains data from 2015 up to 2022. the datasets used contain information from the Gallup World Survey. This project will focus on the data from 2019 and 2020. I am interested in comparing how the Covid-19 pandemic affected the overall happiness around the world. Each country has a happiness score that is achieved by considering information collected from survey-based research where respondents responded to main life evaluation questions and ranked their current lives on a scale of zero to ten. In every country, a sample size of 2000 to 3000 people was asked to think of a ladder where the best possible life for them would be a ten and the worst possible experience being a zero. Survey respondents answered questions like “If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them?” or “Have you donated money to a charity in the past month?”, or “Are you satisfied by or dissatisfied with your freedom to choose what to do with your life?” These questions helped in determining the level of social support, generosity, and freedom that citizens of different countries have.

### Description of the data 

Each variable measured reveals a populated-weighted average score on a scale running from 0 to 10 that is tracked over time and compared against other countries. These variables currently include real GDP per capita, social support, healthy life expectancy, freedom to make life choices, generosity, and perceptions of corruption. 

GDP: GDP per capita is a measure of the overall domestic production and it functions as a comprehensive scorecard of a given country’s economic health. 

social: Social support means the ability to have family and friends or other people who can help you in time of need. Social support improves happiness in people’s lives because they do not have to worry about being alone in difficult situations.

health: Healthy Life Expectancy is the average years of life in good health. Life without disability illnesses or injuries.

freedom: Freedom of choice describes an individual’s ability to choose what they do with their life. The average of all answers determined the result of every country.

generosity: Respondents were asked whether they have donated money to a charity in the past month. The average of all answers determined the result of every country.

corruption: The Corruption Perceptions Index (CPI) is an index published annually by Transparency International since 1995, which ranks countries “by their perceived levels of public sector corruption, as determined by expert assessments and opinion surveys.”

In some cases where countries are missing one or more happiness factors over the survey period, information from earlier years is used as if they were current information. This may cause some bias in my results but it will not make a huge difference because there is a limit of 3 years for how far back the researchers went in search of those missing values. I believe that the dataset used in this project is good, but not 100% accurate respective to years.

### Data Cleaning
While cleaning the datasets used in this project, I wanted to maintain variables that play an important role in explaining national happiness. For both 2019 and 2020, I kept log GDP per capita, Social support, Healthy life expectancy, Freedom to make life choices, Generosity, and Perception of corruption. Luckily, I did not have any missing values for both datasets. The tables below show what my data looks like.

![data](https://user-images.githubusercontent.com/98835048/167772309-06e657b9-4a7c-40e0-bd38-4a62244dcb67.png)


### Correlation Coefficients for all Numerical variables
I started my analysis by checking the correlation coefficients between all variables to have an idea of what correlates more with the happiness score.  GDP, social support,  and health expectancy have the highest correlation coefficient with the happiness score for both 2019 and 2020. Correlations are useful to get a quick idea about the data.


![heatmap_2019 (1)](https://user-images.githubusercontent.com/98835048/167767733-32c328d2-0640-49ae-9bc5-0a96a44d83fc.png)
![heatmap_2020 (1)](https://user-images.githubusercontent.com/98835048/167767731-0308d4af-1a82-45a2-9915-c47a703bd0bf.png)

Overall Plot of the the correlation coefficients of all variables. 

![all_corr](https://user-images.githubusercontent.com/98835048/168405000-bf01b85b-5379-4e19-a8d6-fa95b509fa3e.png)


After obtaining the correlation coefficients for all numerical values in my dataset, I used Tableau to obtain correlation visualizations to see how all independent features relate to our dependent feature for every country. 


<img width="500" alt="gdp health" src="https://user-images.githubusercontent.com/98835048/167769134-b745261d-591d-4040-b5e2-f911f88cca9f.png">
<img width="500" alt="Social freedom" src="https://user-images.githubusercontent.com/98835048/167769144-1e9a8d2f-6cc0-4337-af67-cb5fa880c57a.png">
<img width="500" alt="generosity corrupt" src="https://user-images.githubusercontent.com/98835048/167769151-afda2df4-72b5-4f67-977b-e21f98afb2a8.png">

### Description of all methods applied

## Random Forest
Random Forest is a supervised learning algorithm that uses ensemble learning method for classification and regression. Random Forest is one of the most accurate learning algorithms available and it runs very efficiently on large datasets. Random forest consists of a collection of tree–structured classifiers that are independent, identically distributed random vectors and each tree casts a unit vote for the most popular class at input x . Random forests for regression are formed by planting trees depending on the random variable x , relative to each category label, tree predictor gives a numerical result. 
Random Forest as a combination of the tree classifer is an effective classification predicting tool. Random forest has a high accuracy level and it runs faster compared to other methods. In addition, it does not produce over-fitting. Random forest also makes it easy to find generalization error, correlation and strength, can also estimate the importance of individual variables. Random forest can also handle continuous variables and categorical variables. 

### Support Vector Machine (SVM)
SVMs are a set of supervised learning methods used for classification, regression, and outliers detection. The SVM classifiers work for both linear and nonlinear classes of data through Kernel tricks. A Support Vector Machine is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data, the algorithm outputs an optimal hyperplane that categorizes new samples. Classification is the task of choosing the correct class label for given data input. In basic classification tasks, each input is considered in isolation from all other inputs, and the set of labels is defined in advance. Classification can be challenging because data is often of high dimension.

An SVM training algorithm builds a model of data points in space so that the data points of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall on. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using the kernel trick, which implicitly maps their inputs into high-dimensional feature spaces. More formally, an SVM constructs a hyperplane or set of hyperplanes in a high or infinite-dimensional space, which can be used for classification, regression, or other tasks.

### Locally Weighted Regression
Locally weighted regression (LWR) is a memory-based method that performs a regression around a point of interest using only training data that are ``local'' to that point. The LOESS model performs a linear regression on points in the data set, weighted by a kernel centered at x. The kernel shape is a design parameter for which there are many possible choices. 

### Libraries used in my project

```Python

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import xgboost as xgb

```
## Application of our models on the World Happiness Report data from 2019

### Splitting the data into test and train datasets and standardize the input features.

```Python
X = data_2019.drop(['score','country'], axis = 1).values
y = data_2019['score'].values

# split the data into a train/test set andstandardize
xtrain, xtest, ytrain, ytest = tts(X,y,test_size=0.25, random_state=123)

```
### importing kernels

```Python
# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 

```

### Defining Kernel regression model
```Python
#Defining the kernel local regression model

def lw_reg(X, y, xnew, kern, tau, intercept):
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) 
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
```

```Python
def boosted_lwr(X, y, xnew, kern, tau, intercept):
  Fx = lw_reg(X,y,X,kern,tau,intercept)
  # Now train the Decision Tree on y_i - F(x_i)
  new_y = y - Fx
  model = RandomForestRegressor(n_estimators=100,max_depth=2)
  model.fit(X,new_y)
  output = model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output 

```

```Python
# we want more nested cross-validations

mse_lwr = []
mse_blwr = []
mse_rf = []
mse_xgb = []

for i in [1234]:
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    dat_train =np.concatenate([xtrain,ytrain.reshape(-1,1)], axis =1)
    dat_test= np.concatenate([xtest, ytest.reshape(-1,1)], axis = 1)

    yhat_lwr = lw_reg(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
    yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
    model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
    model_rf.fit(xtrain,ytrain)
    yhat_rf = model_rf.predict(xtest)
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)

    mse_lwr.append(mse(ytest,yhat_lwr))
    mse_blwr.append(mse(ytest,yhat_blwr))
    mse_rf.append(mse(ytest,yhat_rf))
    mse_xgb.append(mse(ytest,yhat_xgb))

print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for BLWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Squared Error for RF is : '+str(np.mean(mse_rf)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))

```
results:

The Cross-validated Mean Squared Error for LWR is : 0.4443231257009582

The Cross-validated Mean Squared Error for BLWR is : 0.45219311571557935

The Cross-validated Mean Squared Error for RF is : 0.27977824363782855

The Cross-validated Mean Squared Error for XGB is : 0.43598706049177594

## Kernel Regression
``` Python
from statsmodels.nonparametric.kernel_regression import KernelReg
dat = np.concatenate([X,y.reshape(-1,1)],axis=1)
model_KernReg = KernelReg(endog=dat[:,-1],exog=dat[:,:-1],var_type='cccccc',ckertype='gaussian')
yhat_sm, y_std = model_KernReg.fit(dat[:,:-1])
mse(y,yhat_sm)
```

results:

mse = 0.11100890996144133


### Lasso
```Python
model_lasso = Lasso(alpha = 0.1, fit_intercept= False)
model_lasso.fit(X,y)
model_lasso.coef_

```
results:

array([0.26868108, 3.81776062, 0.4927256 , 0.        , 0.        ,
       0.        ])
       
### Ridge
```Python
model_ridge = Ridge(alpha = 0.1, fit_intercept = False)
model_ridge.fit(X,y)
model_ridge.coef_
```
results:

array([0.25709581, 2.30327954, 1.64660648, 1.74019488, 1.91512056,
       0.74932409])
      
Ridge regression puts a constraint on the coefficients. The penalty term (lambda) regularizes the coefficients such that if the coefficients take large values the optimization function is penalized. So, ridge regression shrinks the coefficients and it helps to reduce the model complexity and multi-collinearity.  Lasso is similar to ridge regression cost function, but the only difference is instead of taking the square of the coefficients, magnitudes are taken into account.  Lasso regression not only helps in reducing over-fitting but it can help us in feature selection

Lasso performed better on both the 2019 dataset and 2020 dataset compared to Ridge. with the same value of alpha, 

### Elastic Net
```Python
model_net = ElasticNet(alpha = 0.2, l1_ratio = 0.5, fit_intercept = False)
model_net.fit(X,y)
model_net.coef_

```
results:

array([1.12078789, 2.49474358, 1.08308824, 0.4946387 , 0.        ,
       0.        ])

### Square Root Lasso

```Python
import statsmodels.api as sm
model_sqrtlasso = sm.OLS(y,X)
result_sqrtlasso = model_sqrtlasso.fit_regularized(method='sqrt_lasso', alpha=2)
result_sqrtlasso.params
```

results: 

array([0.24544518, 2.41175393, 1.64297973, 1.76717266, 1.57950054,
       0.05387389])

## Application of our models on the World Happiness Report data from 2020

The Cross-validated Mean Squared Error for LWR is : 0.5461699734502683

The Cross-validated Mean Squared Error for BLWR is : 0.5540449397777254

The Cross-validated Mean Squared Error for RF is : 0.32746581328778185

The Cross-validated Mean Squared Error for XGB is : 0.49518979091630166

The Cross-validated Mean Squared Error for Kernel Regression is : 0.1506922680827494

### Lasso
array([ 0.194633  ,  0.        ,  0.05725919,  0.        ,  0.        ,
       -0.        ])
       
### Ridge 
array([ 0.21528381,  2.83366005,  0.01993118,  1.12490197,  0.25589257,
       -1.32370785])
       
### Elastic Net

array([ 0.15699204,  0.        ,  0.06268717,  0.        ,  0.        ,
       -0.        ])
       
### Squareroot Lasso

array([ 0.25094054,  1.92372041,  0.02869725,  0.78870295,  0.05962698,
       -1.18627133])





### Model evaluation and comparison
To evaluate different model performances, I compared regression-based R^2, mean square error. R^2 was derived from correlations between predicted and observed values. All models had good overall performance when evaluating the mean squared error for 2019 world happiness data and 2020. Among all models, the random forest had the lowest mean squared error and it performed best on both the 2019 data and 2020 data. I applied all models to the data collected during the year of 2020, and the accuracy of all models declined. The covid-19 pandemic negatively affected the data collection process and it affected the quality of the whole dataset, hence poor predictions by our models.

### References 

World Happiness Report up to 2022. (n.d.). Retrieved May 10, 2022, from https://www.kaggle.com/mathurinache/world-happiness-report

Inc, G. (n.d.). Tracking the World’s Happiness. Gallup.Com. Retrieved May 11, 2022, from https://www.gallup.com/analytics/247355/gallup-world-happiness-report.aspx

Bhattacharyya, S. (2020, September 28). Ridge and Lasso Regression: L1 and L2 Regularization. Medium. https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b

Cleveland, W. S., & Devlin, S. J. (1988). Locally Weighted Regression: An Approach to Regression Analysis by Local Fitting. Journal of the American Statistical Association, 83(403), 596–610. https://doi.org/10.1080/01621459.1988.10478639

Locally Weighted Regression. (n.d.). Retrieved May 13, 2022, from http://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/cohn96a-html/node7.html

Liu, B., Ma, M., & Chang, J. (Eds.). (2012). Information Computing and Applications (Vol. 7473). Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-642-34062-8

Gross Domestic Product (GDP) Definition. (n.d.). Investopedia. Retrieved May 10, 2022, from https://www.investopedia.com/terms/g/gdp.asp

Gross Domestic Product (GDP) Definition. (n.d.). Investopedia. Retrieved May 10, 2022, from https://www.investopedia.com/terms/g/gdp.asp

Amarappa, S., & Sathyanarayana, D. S. V. (n.d.). Data classification using Support vector Machine (SVM), a simplified approach. 11.

Chen, J., de Hoogh, K., Gulliver, J., Hoffmann, B., Hertel, O., Ketzel, M., Bauwelinck, M., van Donkelaar, A., Hvidtfeldt, U. A., Katsouyanni, K., Janssen, N. A. H., Martin, R. V., Samoli, E., Schwartz, P. E., Stafoggia, M., Bellander, T., Strak, M., Wolf, K., Vienneau, D., … Hoek, G. (2019). A comparison of linear regression, regularization, and machine learning algorithms to develop Europe-wide spatial models of fine particles and nitrogen dioxide. Environment International, 130, 104934. https://doi.org/10.1016/j.envint.2019.104934

García-Portugués, E. (n.d.). 5.3 Inference for model parameters | Notes for Predictive Modeling. Retrieved May 13, 2022, from https://bookdown.org/egarpor/PM-UC3M/glm-inference.html




