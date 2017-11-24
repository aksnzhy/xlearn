## Python Package Usage

Here we demonstrate how to use xLearn python package. Make sure that you have already installed the shared libary (`libxlearn.so` for linux and `libxlearn.dylib` for Mac OSX), and export the python path to your environment. See this page [install.md][1] for installation.

The training data - `small_train.txt` and testing data `small_test.txt` (in the root directory of your xlearn build package) is a portion of the whole data of criteo ctr prediction challenge in [kaggle][2].

#### Quick start

    # coding: utf-8
    # This file test the xlearn python package.
    # We create a ffm model for binary classification problem.
    # The dataset comes from the criteo CTR.
    import xlearn as xl
    
    # Create factorazation machine
    ffm_model = xl.create_ffm()
    
    # Set training data and validation data
    ffm_model.setTrain("./small_train.txt")
    ffm_model.setValidate("./small_test.txt")
    
    # Set hyper-parameters
    param = { 'task':'binary',
              'lr' : 0.2,
              'lambda' : 0.002,
              'metric' : 'auc' }
    
    # Tarin model
    ffm_model.fit(param, "model.out")
    
    # Predict
    ffm_model.setTest("./small_test.txt")
    ffm_model.predict("model.out", "output")

 The above code shows how to use xLearn python API to train model and make prediction. In this example, we use ffm to solve the binary classification problem.

#### Create model



  [1]: install.md
  [2]: https://www.kaggle.com/c/criteo-display-ad-challenge