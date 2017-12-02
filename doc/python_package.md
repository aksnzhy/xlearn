## Python Package Usage

Here we demonstrate how to use xLearn python package. Make sure that you have already installed the shared libary (`libxlearn.so` for linux and `libxlearn.dylib` for Mac OSX), and export the python path to your environment. See this page [install.md][1] for installation.

The training data - `small_train.txt` and testing data `small_test.txt` (in the root directory of your xlearn build package) is a portion of the whole data of criteo ctr prediction challenge in [kaggle][2].

#### Data format

For now, xLearn can support three data format, including `libsvm`, `libffm`, and `csv`. 

    libsvm : 
       y1 idx:value idx:value ...
       y2 idx:value idx:value ...
    
    libffm:
       y1 field:idx:value field:idx:value ...
       y2 field:idx:value field:idx:value ...
    
    csv:
       feat_1 feat_2 feat_3 ... feat_n y1
       feat_1 feat_2 feat_3 ... feat_n y2

Note that the CSV format can only be used in linear and fm model. For ffm, users need to convert their data to libffm format.
**Also, when using csv, users need to add a dummy `y` at the end of the test data in every line.**

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

We can creat a model by using `xlearn.create_xxx()` function:

    # Create linear model
    create_linear()
    # Create factorization machine
    create_fm()
    # Create field-aware factorization machine
    create_ffm()

#### Set data

We can set training data, validation data, and testing data by using the following functions:

    setTrain("file_name")
    setValidate("file_name")
    setTest("file_name")

#### Hyper-parameters

We can set hyper-parameters to xLearn by using a python `dictionary`:

    param = { 'task':'binary',   # ‘reg’ for regression task
              'metric':'auc',
              'lr':0.2,
              'k':4,
              'lambda':0.0002
              'init':0.66,
              'epoch':10,
              'fold':5}

#### The other set funtions

We can also use some other set funtions:

    setQuiet()
    disableNorm()
    disableLockFree()
    disableEarlyStop()
    disableSign()
    disableSigmoid()

#### Train model

We can train our model by using `fit()` function:

    fit(param, "model_output")

#### Cross-Validation

We can also perform cross-validation by using `cv()` function:

    cv(param)

#### Predict

We can perform prediction by using `predict()` function:

    predict("model_file", "output_file")

#### Set Output format

On default, xlearn will only output the score for prediction. If you want to get a result between 0~1. You can use `setSigmoid()` function.

If you want to convert the output to a binary result, i.e., 0 (false) or 1 (true), You can use `setSign` function.

  [1]: install.md
  [2]: https://www.kaggle.com/c/criteo-display-ad-challenge
