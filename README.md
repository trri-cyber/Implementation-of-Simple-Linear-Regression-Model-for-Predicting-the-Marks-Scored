# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Rishab p doshi
RegisterNumber:  212224240134
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
```
```
#reading csv file
df= pd.read_csv('/content/student_scores.csv')
```
```
#displaying the content in datafile
print(df.head())
print(df.tail())
```
```
# Segregating data to variables-X
X=df.iloc[:,:-1].values
print(X)
```
```
# Segregating data to variables-Y
y=df.iloc[:,-1].values
print(y
```
```
#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)
```
```
#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
```
```
#displaying predicted values
y_pred=regressor.predict(X_test)
y_pred
```
```
#displaying actual values
y_test
```
```
#graph plot for training data
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
```
```
#graph plot for test data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='green')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
```
```
#find mae,mse,rmse
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

Content in head and tail of datafile

![image](https://github.com/user-attachments/assets/f423763c-8b4d-40ba-b1a0-c5f29c9ec5ef)

X-VALUES

![image](https://github.com/user-attachments/assets/9ff81765-f484-4c3f-a7fa-22971cd9dd84)

Y-VALUES

![image](https://github.com/user-attachments/assets/0599de54-8669-4f45-9b1a-1363f179ee4a)

PREDICTED VALUES OF Y

![image](https://github.com/user-attachments/assets/7c079ffd-e7be-47ee-9a2f-b3df09e9f9c8)

ACTUAL VALUES OF Y

![image](https://github.com/user-attachments/assets/4623d4ad-8291-4d87-8661-2a8f42cbe01d)

GRAPH FOR TRAINING SET

![image](https://github.com/user-attachments/assets/262b69e3-0499-4e73-b7f5-7e6bb6b25fd5)

GRAPH FOR TESTING SET

![image](https://github.com/user-attachments/assets/dcf2d6d4-0432-4812-869a-bef3ed3b4b8b)

MSE,MAE,RMSE VALUES

![image](https://github.com/user-attachments/assets/157628ae-2824-44b2-a6d6-6a53024eb0fa)









## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
