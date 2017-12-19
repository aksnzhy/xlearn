import xlearn as xl

# Training task
linear_model = xl.create_linear()
linear_model.setTrain("./agaricus_train.txt")
linear_model.setValidate("./agaricus_test.txt")
param = {'task':'binary', 'lr':0.2, 
         'lambda':0.002, 'metric':'acc', 
         'opt':'sgd'}

linear_model.fit(param, './model.out')

# Prediction task
linear_model.setTest("./agaricus_test.txt")
# Convert output to 0-1
linear_model.setSigmoid()
linear_model.predict("./model.out", "./output.txt")