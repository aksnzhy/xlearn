import numpy as np
import xlearn as xl
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
wine_data = load_wine()
X = wine_data['data']
y = (wine_data['target'] == 1)

X_train,    \
X_val,      \
y_train,    \
y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize input
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# param:
#  0. binary classification
#  1. model scale: 0.1
#  2. epoch number: 10 (auto early-stop)
#  3. number of latent factor: 4
#  4. learning rate: 0.1
#  5. regular lambda: 0.01
#  6. use sgd optimization method
#  7. evaluation metric: accuarcy
fm_model = xl.FMModel(task='binary', init=0.1, 
                      epoch=10, k=4, lr=0.1, 
                      reg_lambda=0.01, opt='sgd', 
                      metric='acc')
# Start to train
fm_model.fit(X_train, 
             y_train, 
             eval_set=[X_val, y_val])

# print model weights
print(fm_model.weights)

# Generate predictions
y_pred = fm_model.predict(X_val)