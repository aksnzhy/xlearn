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

# coding: utf-8
# This file test the xlearn python package.
# We create a ffm model for binary classification problem.
# The dataset comes from the criteo CTR.
from __future__ import absolute_import
import xlearn as xl

# Training task
ffm_model = xl.create_ffm() # Use field-aware factorization machine
ffm_model.setTrain("./small_train.txt")  # Training data
ffm_model.setValidate("./small_test.txt")  # Validation data

# param:
#  0. binary classification
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: accuracy
param = {'task':'binary', 'lr':0.2, 
         'lambda':0.002, 'metric':'acc'}

# Start to train
# The trained model will be stored in model.out
ffm_model.fit(param, './model.out')

# Prediction task
ffm_model.setTest("./small_test.txt")  # Test data
ffm_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict("./model.out", "./output.txt")
