import xlearn as xl

# Training task
fm_model = xl.create_fm()  # Use factorization machine
fm_model.setTrain("./titanic_train.txt")  # Training data

# param:
#  0. Binary classification task
#  1. learning rate: 0.2
#  2. lambda: 0.002
#  3. metric: accuracy
param = {'task':'binary', 'lr':0.2, 
         'lambda':0.002, 'metric':'acc'}

# Use cross-validation
fm_model.cv(param)