# coding: utf-8
# This file test the xlearn python package.
# We create a ffm model for binary classification problem.
# The dataset comes from the criteo CTR.
import xlearn as xl

# Create factorazation machine
ffm_model = xl.create_ffm()

# Set training data and validation data
ffm_model.setTrain("small_train.txt")
ffm_model.setValidate("small_test.txt")

# Set hyper-parameters
param = { 'task':'binary',
          'lr' : 0.2,
          'lambda' : 0.002,
          'metric' : 'auc' }

# Tarin model
ffm_model.fit(param, "model.out")

# Predict
ffm_model.setTest("small_test.txt")
ffm_model.predict("model.out", "output")