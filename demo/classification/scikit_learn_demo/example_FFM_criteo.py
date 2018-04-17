import numpy as np
import xlearn as xl

# param:
#  0. binary classification
#  1. learning rate: 0.2
#  2. epoch number: 10 (auto early-stop)
#  3. evaluation metric: accuarcy
#  4. use sgd optimization method
ffm_model = xl.FFMModel(task='binary', 
                        lr=0.2, 
                        epoch=10, 
                        reg_lambda=0.002,
                        metric='acc')
# Start to train
# Directly use string to specify data source
ffm_model.fit('../criteo_ctr/small_train.txt', 
              eval_set='../criteo_ctr/small_test.txt')

# print model weights
print(ffm_model.weights)

# Generate predictions
y_pred = ffm_model.predict('../criteo_ctr/small_test.txt')
