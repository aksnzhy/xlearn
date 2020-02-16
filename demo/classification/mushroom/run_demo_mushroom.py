# Copyright (c) 2018 by contributors. All Rights Reserved.
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

import xlearn as xl

# Training task
linear_model = xl.create_linear()  # Use linear model
linear_model.setTrain("./agaricus_train.txt")  # Training data
linear_model.setValidate("./agaricus_test.txt")  # Validation data

# param:
#  0. binary classification
#  1. learning rate: 0.2
#  2. lambda: 0.002
#  3. evaluation metric: accuracy
#  4. use sgd optimization method
param = {'task':'binary', 'lr':0.2, 
         'lambda':0.002, 'metric':'acc', 
         'opt':'sgd'}

# Start to train
# The trained model will be stored in model.out
linear_model.fit(param, './model.out')

# Prediction task
linear_model.setTest("./agaricus_test.txt")  # Test data
linear_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
linear_model.predict("./model.out", "./output.txt")