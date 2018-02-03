from sklearn.datasets import load_iris
from xlearn.sklearn import FMModel

# load dataset
iris_data = load_iris()
X = iris_data['data']
y = (iris_data['target']==2)

# initialize and fit model
mdl = FMModel(task='binary', init=0.1, epoch=1, k=1, lr=0.01, reg_lambda=0.02)
mdl.fit(X, y)

# generate predictions
y_pred = mdl.predict(X)
print(y_pred)
