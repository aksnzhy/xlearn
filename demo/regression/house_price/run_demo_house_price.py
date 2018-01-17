import xlearn as xl

# Training task
ffm_model = xl.create_fm()  # Use factorization machine
ffm_model.setTrain("./house_price_train.txt")  # Training data

# param:
#  0. Binary task
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  4. evaluation metric: rmse
param = {'task':'reg', 'lr':0.2, 
         'lambda':0.002, 'metric':'rmse'}

# Use cross-validation
ffm_model.cv(param)