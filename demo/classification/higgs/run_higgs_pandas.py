# Copyright (c) 2019 by contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This is a snippet to test DMatrix transition from pandas DataFrame
# Note: this program modify by demo of higgs,
# If users want to check DMatrix work well, set disable lock-free in
# both this file and run_higgs.py, there maybe some difference of precision
# with output result file

import xlearn as xl
import numpy as np
import pandas as pd

# read file from file
higgs_train = pd.read_csv("higgs-train.csv", header=None, sep=",")
higgs_test = pd.read_csv("higgs-test.csv", header=None, sep=",")

# get train X, y
X_train = higgs_train[higgs_train.columns[1:]]
y_train = higgs_train[0]

# get test X, y
X_test = higgs_test[higgs_test.columns[1:]]
y_test = higgs_test[0]

# DMatrix transition
xdm_train = xl.DMatrix(X_train, y_train)
xdm_test = xl.DMatrix(X_test, y_test)

# Training task
linear_model = xl.create_linear()  # Use linear model
# we use the same API for train from file
# that is, you can also pass xl.DMatrix for this API now
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
# we use the same API for test from file
# that is, you can also pass xl.DMatrix for this API now
linear_model.setTest(xdm_test)  # Test data
linear_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
# if no result out path setted, we return res as numpy.ndarray
res = linear_model.predict("./model_dm.out")

print(res)
