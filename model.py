
# Deployment of ML models in Heroku using Flask

# heruko - PAAS

# Things to do :- 
# 1) Train our Model
# 2) Create the Web App using Flask 
# 3) Commit the code in Github 
# 4) Create an Account in Heroku(PAAS)
# 5) Link the Github to Heroku
# 6) Deploy the model 
# 7) Web app is ready 


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle 

data = pd.read_csv("D:\ML_Practice\Model Deployment_Krish Naik\hiring.csv")
data['experience'].fillna('zero',inplace=True)
data['test_score'].fillna(data['test_score'].mean(),inplace=True)

from sklearn.preprocessing import LabelEncoder
label_exp = LabelEncoder()
data['experience'] = label_exp.fit_transform(data['experience'])

X= data.iloc[:,:3]
y= data.iloc[:,-1]

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X,y)

# saving model to disk 
pickle.dump(regressor,open('model.pkl','wb')) 