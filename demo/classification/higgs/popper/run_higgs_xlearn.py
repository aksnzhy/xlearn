# Import dataset
import numpy as np
import pandas as pd
import xlearn as xl
from sklearn.model_selection import train_test_split

# Load dataset
higgs = pd.read_csv("HIGGS.csv", header=None, sep=",")

X = higgs[higgs.columns[1:]]
y = higgs[0]

# Split train and test set 
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# DMatrix transition
xdm_train = xl.DMatrix(x_train, y_train)
xdm_test = xl.DMatrix(x_test, y_test)

# Training task
linear_model = xl.create_linear()  # Use linear model
linear_model.setTrain(xdm_train)    # Training data
linear_model.setValidate(xdm_test)  # Validation data

# param:
#  0. regression task
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: acc
param = {'task':'binary', 'lr':0.2, 
         'lambda':0.002, 'metric':'acc'}

# Start to train
# The trained model will be stored in model.out
linear_model.fit(param, './model_dm.out')

# Prediction task
linear_model.setTest(xdm_test)  # Test data
linear_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
# if no result out path setted, we return res as numpy.ndarray
res = linear_model.predict("./model_dm.out")

print(res)

