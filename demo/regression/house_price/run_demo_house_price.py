import xlearn as xl

# Training task
ffm_model = xl.create_ffm()
ffm_model.setTrain("./house_price_train.txt")
param = {'task':'reg', 'lr':0.2, 
         'lambda':0.002, 'metric':'rmse'}

# Cross-validation
ffm_model.cv(param)