'''
Bussiness Problem :	A logistics company recorded the time taken for delivery and the time taken for the sorting of the items for delivery. 
Build a Simple Linear Regression model to find the relationship between delivery time and sorting time with the delivery time 
as the target variable. Apply necessary transformations and record the RMSE and correlation coefficient values for different models

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline  
from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf  # Statsmodels for statistical modeling

data = pd.read_csv(r"C:\Users\User\Desktop\DATA SCIENCE\ds_assignment\7) simple_linear_regression\delivery_time.csv")

##  EDA

data.describe()
data.info()
data.head()
data.columns


# split data

X = pd.DataFrame(data['SortingTime'])  
Y = pd.DataFrame(data['DeliveryTime'])  


numeric_feature = ['SortingTime']

num_pipleline = Pipeline([
            ('imputer', SimpleImputer(strategy = 'median')),
            ('winsor', Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5))
    ]) 


# 1st MBD

data['DeliveryTime'].mean()
data['DeliveryTime'].median()
data['DeliveryTime'].mode() 

data['SortingTime'].mean()
data['SortingTime'].median()
data['SortingTime'].mode()

# 2nd MBD

data['DeliveryTime'].var()
data['DeliveryTime'].std()
range1 = max(data['DeliveryTime']) - min(data['DeliveryTime'])

data['SortingTime'].var()
data['SortingTime'].std()
range2 = max(data['SortingTime']) - min(data['SortingTime'])

 
# Data Visualization


# distplot


sns.distplot(data['DeliveryTime'])
plt.show()

sns.distplot(data['SortingTime'])
plt.show()


# boxplot

plt.boxplot(data['DeliveryTime'])
plt.title('DeliveryTime')
plt.show()

plt.boxplot(data['SortingTime'])
plt.title('SortingTime')
plt.show()

plt.scatter(data['DeliveryTime'], data['SortingTime'])
plt.show()

# 3rd MBD

data['DeliveryTime'].skew()
data['SortingTime'].skew()


# 4th MBD

data['DeliveryTime'].kurt()
data['SortingTime'].kurt()



correlation = np.corrcoef(data['DeliveryTime'], data['SortingTime'])[0, 1]
print(correlation)


covariance = np.cov(data['DeliveryTime'], data['SortingTime'])
print(covariance)

# EDA completed

# Data Preprocessing


duplicate = data.duplicated(keep = 'last')
sum(duplicate)


''' 
Data type is correct,no duplicates, no missing values are there, no need to scaling, no null values are present

'''


# Model Bulding   

# Simple Linear Regression Model

model = smf.ols('DeliveryTime ~ SortingTime', data = data).fit()

print('Linear Regression Model Summary : \n')
print(model.summary())

pred1 = model.predict(pd.DataFrame(data['SortingTime'])) 


# Error calculation

res1 = data.DeliveryTime - pred1
res_sqr1 = res1 * res1
mse = np.mean(res_sqr1)
rmse = np.sqrt(mse)

print('Root Mean square Error (RMSE) for Base Model : ', rmse)


plt.scatter(data.SortingTime, data.DeliveryTime)
plt.plot(data.SortingTime, pred1, 'r')
plt.xlabel('SortingTime')
plt.ylabel('DeliveryTime')
plt.title('Linear regression Line (DeliveryTime ~ SortingTime)')
plt.legend(['Observed data', 'fitted line'])
plt.show()


# Model Tuning with Transformations
# 1. Log Transformation of Predictor Variable

plt.scatter(x = np.log(data.DeliveryTime), y = data.SortingTime, color = 'brown' )
plt.xlabel('Log(Delivery Time)')
plt.ylabel('Sorting Time')
plt.title('Scatter Plot with Logtransformed Sorting Time')
plt.show()

print('Correlation after Log Transform : ', np.corrcoef(np.log(data.SortingTime), data.DeliveryTime)[0,1])



# Fit Linear Regression with Log-Transformed Predictor

model2 = smf.ols('SortingTime ~ np.log(DeliveryTime)', data = data).fit() 
print('Model Summary for Log-transformed model : \n')
print(model2.summary())


pred2 = model2.predict(pd.DataFrame(data['DeliveryTime']))


res2 = data['DeliveryTime'] - pred2
res2_sqr = res2 * res2
mse2 = np.mean(res2_sqr)
rmse2 = np.sqrt(mse)

print('RMSE for Log-transformed model : ',rmse2)




plt.scatter(x = np.log(data['DeliveryTime']), y = data.SortingTime)
plt.plot(np.log(data['DeliveryTime']), pred2, 'r')
plt.xlabel('Log(Delivery Time)')
plt.ylabel('Sorting Time')
plt.title('Regression Line with Log-transformed Delivery Time')
plt.legend('Observed data', 'fitted line')
plt.show()



# Fit Linear Regression with Exponential-Transformed Response

model3 = smf.ols('np.log(DeliveryTime) ~ SortingTime', data = data).fit() 
print("Model Summary for exponential-transformed model:") 
print(model3.summary()) 

pred3 = model3.predict(pd.DataFrame(data['SortingTime']))  
# Error Calculation for Exponential-Transformed Model
pred3_at = np.exp(pred3) 
res3 = data['DeliveryTime'] - pred3_at 
res_sqr3 = res3 * res3  
mse3 = np.mean(res_sqr3) 
rmse3 = np.sqrt(mse3)  
print("RMSE for exponential-transformed model:", rmse3) 

# Predictions and Visualization for Exponential-Transformed Model
plt.scatter(data['SortingTime'], np.log(data.DeliveryTime))  
plt.plot(data.SortingTime, pred3, "r")  
plt.xlabel('Sorting Time') 
plt.ylabel('Log(Delivery Time)') 
plt.title('Regression Line with Exponential-Transformed Delivery Time') 
plt.legend(['Observed data', 'Predicted line']) 
plt.show() 

# Comparing Model Performance
print("The base model has a RMSE of:", rmse)  # Print RMSE of the base model
print("The log-transformed model has a RMSE of:", rmse2)  # Print RMSE of the log-transformed model
print("The exponential-transformed model has a RMSE of:", rmse3)  # Print RMSE of the exponential-transformed model


'''
Conclusion:
    
    Exponential-transformed model RMSE is low so we will go for model3 for Simple Linear Regression.
'''

