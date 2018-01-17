import xlearn as xl

# Training task
linear_model = xl.create_linear()  # Use linear model
linear_model.setTrain("./agaricus_train.txt")  # Training data
linear_model.setValidate("./agaricus_test.txt")  # Validation data

# param:
#  0. Binary classification
#  1. learning rate: 0.2
#  2. lambda: 0.002
#  3. evaluation metric: accuarcy
#  4. Use sgd optimization method
param = {'task':'binary', 'lr':0.2, 
         'lambda':0.002, 'metric':'acc', 
         'opt':'sgd'}

# Start to train
linear_model.fit(param, './model.out')

# Prediction task
linear_model.setTest("./agaricus_test.txt")  # Test data
linear_model.setSigmoid()  # Convert output to 0-1

# Start to predict
linear_model.predict("./model.out", "./output.txt")