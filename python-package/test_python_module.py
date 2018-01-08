#!/usr/bin/python
# coding: utf-8
# This file test the xlearn python package.
# We create a ffm model for binary classification problem.
# The dataset comes from the criteo CTR.
from __future__ import absolute_import
import numpy as np
import xlearn as xl

from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file

# Set hyper-parameters
param = { 'task':'binary',
          'lr' : 0.2,
          'lambda' : 0.002,
          'metric' : 'acc' }

X, Y = load_svmlight_file("./test_dmatrix.txt")
print(type(X), type(Y))
print(Y.dtype)
tmp_dmatrix = xl.DMatrix(X, Y)
fm_model = xl.create_fm()
fm_model.setTrain(tmp_dmatrix)
fm_model.setValidate(tmp_dmatrix)
fm_model.fit(param, "fm_model.out")


# Create factorazation machine
ffm_model = xl.create_ffm()

# Set training data and validation data
dtrain = xl.DMatrix("./small_train.txt")
dtest = xl.DMatrix("./small_test.txt")
#ffm_model.setTrain("./small_train.txt")
ffm_model.setTrain(dtrain)
ffm_model.setValidate(dtest);
#ffm_model.setValidate("./small_test.txt")

# Tarin model
ffm_model.fit(param, "model.out")

# Predict
#ffm_model.setTest("./small_test.txt")
ffm_model.setTest(dtest)
ffm_model.predict("model.out", "output")


