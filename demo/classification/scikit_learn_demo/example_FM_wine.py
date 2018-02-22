import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xlearn.sklearn import FMModel

# load dataset
wine_data = load_wine()
X = wine_data['data']
y = (wine_data['target'] == 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# standardize input
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# initialize and fit model
print('Testing FMModel')
mdlFM = FMModel(task='binary', init=0.1, epoch=10, k=4, lr=0.1, reg_lambda=0.01, opt='sgd', metric='acc')
mdlFM.fit(X_train, y_train, eval_set=[X_val, y_val])

# generate predictions
y_pred = mdlFM.predict(X_val)
