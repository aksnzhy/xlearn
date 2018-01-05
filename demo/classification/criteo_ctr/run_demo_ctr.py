import xlearn as xl

# Training task
ffm_model = xl.create_ffm()
ffm_model.setTrain("./small_train.txt")
ffm_model.setValidate("./small_test.txt")
param = {'task':'binary', 'lr':0.2, 
         'lambda':0.002, 'metric':'acc'}

ffm_model.fit(param, './model.out')

# Prediction task
ffm_model.setTest("./small_test.txt")
# Convert output to 0-1
ffm_model.setSigmoid()
ffm_model.predict("./model.out", "./output.txt")