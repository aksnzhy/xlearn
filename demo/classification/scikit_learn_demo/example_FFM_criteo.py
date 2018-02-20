import numpy as np
from xlearn.sklearn import FFMModel

# initialize and fit model
print('Testing FFMModel')
mdlFFM = FFMModel(task='binary', lr=0.2, epoch=10, reg_lambda=0.002, metric='acc')

# directly use string to specify data source
mdlFFM.fit('../criteo_ctr/small_train.txt', eval_set='../criteo_ctr/small_test.txt')

# generate predictions
y_pred = mdlFFM.predict('../criteo_ctr/small_test.txt')
