import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xlearn.sklearn import LRModel

# load dataset
iris_data = load_iris()
X = iris_data['data']
y = (iris_data['target'] == 2)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

# initialize and fit model
print('Testing LRModel')
mdlLR = LRModel(task='binary', init=0.1, epoch=10, lr=0.1, reg_lambda=1, opt='sgd')
mdlLR.fit(X_train, y_train, eval_set=[X_val, y_val], is_lock_free=False)

# generate predictions
y_pred = mdlLR.predict(X_val)
