import xlearn as xl

# Training task
fm_model = xl.create_fm()
fm_model.setTrain("./titanic_train.txt")
param = {'task':'binary', 'lr':0.2, 
         'lambda':0.002, 'metric':'acc'}

# Cross-validation
fm_model.cv(param)