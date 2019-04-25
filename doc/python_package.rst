xLearn Python Package Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^

xLearn supports easy-to-use Python API for users. Once you install the 
xLearn Python package successfully, you can try it. Type ``python`` in your
shell and use the following Python code to check your installation: ::

    import xlearn as xl
    xl.hello()

If you install xLearn Python package successfully, you will see: ::

  -------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.44 Version --
  -------------------------------------------------------------------------

Quick Start
----------------------------------------

Here is a simple Python demo to show that how to use xLearn python API. You can checkout the 
demo data (``small_train.txt`` and ``small_test.txt``) from the path ``demo/classification/criteo_ctr``.

.. code-block:: python

   import xlearn as xl

   # Training task
   ffm_model = xl.create_ffm()                # Use field-aware factorization machine (ffm)
   ffm_model.setTrain("./small_train.txt")    # Path of training data

   # param:
   #  0. task: binary classification
   #  1. learning rate : 0.2
   #  2. regular lambda : 0.002
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002}
            
   # Train model
   ffm_model.fit(param, "./model.out")  

A portion of the xLearn's output: ::
  
   ...
 [ ACTION     ] Start to train ...
 [------------] Epoch      Train log_loss     Time cost (sec)
 [   10%      ]     1            0.595881                0.00
 [   20%      ]     2            0.538845                0.00
 [   30%      ]     3            0.520051                0.00
 [   40%      ]     4            0.504366                0.00
 [   50%      ]     5            0.492811                0.00
 [   60%      ]     6            0.483286                0.00
 [   70%      ]     7            0.472567                0.00
 [   80%      ]     8            0.465035                0.00
 [   90%      ]     9            0.457047                0.00
 [  100%      ]    10            0.448725                0.00
 [ ACTION     ] Start to save model ...

In this example, xLearn uses *feild-ware factorization machines* (ffm) for solving a binary 
classification task. If you want train a model for regression task, you can reset the ``task`` 
parameter to ``reg``: ::

    param = {'task':'reg', 'lr':0.2, 'lambda':0.002} 

We can see that a new file called ``model.out`` has been generated in the current directory. 
This file stores the trained model checkpoint, and we can use this model file to make a prediction 
in the future: ::

    ffm_model.setTest("./small_test.txt")
    ffm_model.predict("./model.out", "./output.txt")      

After we run this Python code, we can get a new file called ``output.txt`` in current directory. 
This is output prediction. Here we show the first five lines of this output by using the following command: ::

    head -n 5 ./output.txt

    -1.58631
    -0.393496
    -0.638334
    -0.38465
    -1.15343

These lines of data are the prediction score calculated for each example in the test set. The negative data 
represents the negative example and positive data represents the positive example. In xLearn, you can convert 
the score to (0-1) by using ``setSigmoid()`` method: ::

   ffm_model.setSigmoid()
   ffm_model.setTest("./small_test.txt")  
   ffm_model.predict("./model.out", "./output.txt")      

and then we can get the result ::

   head -n 5 ./output.txt

  0.174698
  0.413642
  0.353551
  0.414588
  0.250373

We can also convert the score to binary result ``(0 and 1)`` by using ``setSign()`` method: ::

   ffm_model.setSign()
   ffm_model.setTest("./small_test.txt")  
   ffm_model.predict("./model.out", "./output.txt")

and then we can get the result ::

   head -n 5 ./output.txt

   0
   0
   0
   0
   0

Model Output
----------------------------------------

Also, users can save the model in ``TXT`` format by using ``setTXTModel()`` method. For example: ::

    ffm_model.setTXTModel("./model.txt")
    ffm_model.fit(param, "./model.out")

After that, we get a new file called ``model.txt``, which stores the trained model in ``TXT`` format: ::

  head -n 5 ./model.txt

  -1.041
  0.31609
  0
  0
  0

For the linear and bias term, we store each parameter in each line. For FM and FFM, 
we store each vector of the latent factor in each line.  For example:

Linear: ::

  bias: 0
  i_0: 0
  i_1: 0
  i_2: 0
  i_3: 0

FM: ::

  bias: 0
  i_0: 0
  i_1: 0
  i_2: 0
  i_3: 0
  v_0: 5.61937e-06 0.0212581 0.150338 0.222903
  v_1: 0.241989 0.0474224 0.128744 0.0995021
  v_2: 0.0657265 0.185878 0.0223869 0.140097
  v_3: 0.145557 0.202392 0.14798 0.127928

FFM: ::

  bias: 0
  i_0: 0
  i_1: 0
  i_2: 0
  i_3: 0
  v_0_0: 5.61937e-06 0.0212581 0.150338 0.222903
  v_0_1: 0.241989 0.0474224 0.128744 0.0995021
  v_0_2: 0.0657265 0.185878 0.0223869 0.140097
  v_0_3: 0.145557 0.202392 0.14798 0.127928
  v_1_0: 0.219158 0.248771 0.181553 0.241653
  v_1_1: 0.0742756 0.106513 0.224874 0.16325
  v_1_2: 0.225384 0.240383 0.0411782 0.214497
  v_1_3: 0.226711 0.0735065 0.234061 0.103661
  v_2_0: 0.0771142 0.128723 0.0988574 0.197446
  v_2_1: 0.172285 0.136068 0.148102 0.0234075
  v_2_2: 0.152371 0.108065 0.149887 0.211232
  v_2_3: 0.123096 0.193212 0.0179155 0.0479647
  v_3_0: 0.055902 0.195092 0.0209918 0.0453358
  v_3_1: 0.154174 0.144785 0.184828 0.0785329
  v_3_2: 0.109711 0.102996 0.227222 0.248076
  v_3_3: 0.144264 0.0409806 0.17463 0.083712

Online Learning
----------------------------------------
xLearn can supoort online learning, which can train new data based on the pre-trained model. User can use the ``setPreModel`` API to specify the file path of pre-trained model. For example: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")
   ffm_model.setValidate("./small_test.txt")  
   ffm_model.setPreModel("./pre_model")
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

Note that, xLearn can only uses the binary model, not the TXT model.

Choose Machine Learning Algorithm
----------------------------------------

For now, xLearn can support three different machine learning algorithms, including linear model, 
factorization machine (FM), and field-aware factorization machine (FFM): ::
   
    import xlearn as xl

    ffm_model = xl.create_ffm()
    fm_model = xl.create_fm()
    lr_model = xl.create_linear()


For LR and FM, the input data format can be ``CSV`` or ``libsvm``. For FFM, the input data should 
be the ``libffm`` format: ::

  libsvm format:

    label index_1:value_1 index_2:value_2 ... index_n:value_n

  CSV format:

    value_1 value_2 .. value_n label

  libffm format:

    label field_1:index_1:value_1 field_2:index_2:value_2 ...

xLearn can also use ``,`` as the splitor in file. For example: ::

  libsvm format:

     label,index_1:value_1,index_2:value_2 ... index_n:value_n

  CSV format:

     label,value_1,value_2 .. value_n

  libffm format:

     label,field_1:index_1:value_1,field_2:index_2:value_2 ...

Note that, if the csv file doesn’t contain the label ``y``, user should add a ``placeholder`` to the dataset 
by themselves (Also in test data). Otherwise, xLearn will treat the first element as the label ``y``.

In addtion, users can also give a ``libffm`` file to LR and FM task. At that time, 
xLearn will treat this data as ``libsvm`` format. 

Set Validation Dataset
----------------------------------------

A validation dataset is used to tune the hyper-parameters of a machine learning model. In xLearn, users can 
use ``setValdiate()`` API to set the validation dataset. For example: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")
   ffm_model.setValidate("./small_test.txt")  
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

A portion of xLearn's output: ::

  [ ACTION     ] Start to train ...
  [------------] Epoch      Train log_loss       Test log_loss     Time cost (sec)
  [   10%      ]     1            0.589475            0.535867                0.00
  [   20%      ]     2            0.540977            0.546504                0.00
  [   30%      ]     3            0.521881            0.531474                0.00
  [   40%      ]     4            0.507194            0.530958                0.00
  [   50%      ]     5            0.495460            0.530627                0.00
  [   60%      ]     6            0.483910            0.533307                0.00
  [   70%      ]     7            0.470661            0.527650                0.00
  [   80%      ]     8            0.465455            0.532556                0.00
  [   90%      ]     9            0.455787            0.538841                0.00
  [ ACTION     ] Early-stopping at epoch 7

goes down first, and then goes up. This is because the model has already overfitted current training dataset. 
By default, xLearn will calculate the validation loss in each epoch, while users can also set different evaluation 
metrics by using ``-x`` option. For classification problems, the metric can be : ``acc`` (accuracy), ``prec`` (precision), 
``f1`` (f1 score), and ``auc`` (AUC score). For example: ::

   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'acc'}
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'prec'}
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'f1'}
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'auc'}           

For regression problems, the metric can be ``mae``, ``mape``, and ``rmsd`` (rmse). For example: ::

   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'rmse'}
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'mae'}    
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'mape'}  

Cross-Validation
----------------------------------------

Cross-validation, sometimes called rotation estimation, is a model validation technique for assessing how the results 
of a statistical analysis will generalize to an independent dataset. In xLearn, users can use the ``cv()`` API to use 
this technique. For example: ::

    import xlearn as xl

    ffm_model = xl.create_ffm()
    ffm_model.setTrain("./small_train.txt")  
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
    ffm_model.cv(param)

On default, xLearn uses 3-folds cross validation, and users can set the number of fold by 
using the ``fold`` parameter: ::

    import xlearn as xl

    ffm_model = xl.create_ffm()
    ffm_model.setTrain("./small_train.txt")  
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'fold':5} 
            
    ffm_model.cv(param)     

Here we set the number of folds to ``5``. The xLearn will calculate the average validation loss at the 
end of its output message: ::

  [------------] Average log_loss: 0.549758
  [ ACTION     ] Finish Cross-Validation
  [ ACTION     ] Clear the xLearn environment ...
  [------------] Total time cost: 0.05 (sec)

Choose Optimization Method
----------------------------------------

In xLearn, users can choose different optimization methods by using ``opt`` parameter. For now, 
xLearn can support ``sgd``, ``adagrad``, and ``ftrl`` method. By default, xLearn uses the ``adagrad`` method. 
For example: ::

   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'opt':'sgd'} 
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'opt':'adagrad'} 
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'opt':'ftrl'} 

Compared to traditional ``sgd`` method, ``adagrad`` adapts the learning rate to the parameters, performing larger 
updates for infrequent and smaller updates for frequent parameters. For this reason, it is well-suited for dealing 
with sparse data. In addition, ``sgd`` is more sensitive to the learning rate compared with ``adagrad``.

``FTRL`` (Follow-the-Regularized-Leader) is also a famous method that has been widely used in the large-scale sparse 
problem. To use FTRL, users need to tune more hyperparameters compared with ``sgd`` and ``adagrad``.

Hyper-parameter Tuning
----------------------------------------

In machine learning, a hyper-parameter is a parameter whose value is set before the learning process begins. 
By contrast, the value of other parameters is derived via training. Hyper-parameter tuning is the problem of 
choosing a set of optimal hyper-parameters for a learning algorithm.

First, the ``learning rate`` is one of the most important hyperparameters used in machine learning. By default, 
this value is set to ``0.2`` in xLearn, and we can tune this value by using ``lr`` parameter: ::

    param = {'task':'binary', 'lr':0.2} 
    param = {'task':'binary', 'lr':0.5}
    param = {'task':'binary', 'lr':0.01}

We can also use the ``lambda`` parameter to perform regularization. By default, xLearn uses ``L2`` regularization, 
and the *regular_lambda* has been set to ``0.00002``: ::

    param = {'task':'binary', 'lr':0.2, 'lambda':0.01}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.02} 
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 

For the ``FTRL`` method, we also need to tune another four hyper-parameters, 
including ``alpha``, ``beta``, ``lambda_1``, and ``lambda_2``. For example: ::

    param = {'alpha':0.002, 'beta':0.8, 'lambda_1':0.001, 'lambda_2': 1.0}

For FM and FFM, users also need to set the size of latent factor by using ``k`` parameter. By default, 
xLearn uses ``4`` for this value: ::

    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':2}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':4}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':5}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':8}

xLearn uses *SSE* instruction to accelerate vector operation, and hence the time cost 
for ``k=2`` and ``k=4`` are the same.     

For FM and FFM, users can also set the parameter ``init`` for model initialization. 
By default, this value is set to ``0.66``: ::

    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'init':0.80}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'init':0.40}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'init':0.10}
  
Set Epoch Number and Early-Stopping
----------------------------------------

For machine learning tasks, one epoch consists of one full training cycle on the training set. In xLearn, 
users can set the number of epoch for training by using ``epoch`` parameter: ::

    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'epoch':3}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'epoch':5}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'epoch':10}

If you set the validation data, xLearn will perform early-stopping by default. For example: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")
   ffm_model.setValidate("./small_test.txt")
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'epoch':10} 
            
   ffm_model.fit(param, "./model.out") 

Here, we set epoch number to ``10``, but xLearn stopped at epoch ``7`` because we get the best model 
at that epoch (you may get different a stopping number on your local machine): ::

    Early-stopping at epoch 7
    Start to save model ...

Users can set ``window size`` for early-stopping by using ``stop_window`` parameter: ::

    param = {'task':'binary',  'lr':0.2, 
             'lambda':0.002, 'epoch':10,
             'stop_window':3} 
            
    ffm_model.fit(param, "./model.out") 

Users can also disable early-stopping by using ``disableEarlyStop()`` API: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")
   ffm_model.setValidate("./small_test.txt")
   ffm_model.disableEarlyStop();
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'epoch':10} 
            
   ffm_model.fit(param, "./model.out") 

At this time, xLearn performed completed 10 epoch for training.

By default, xLearn will use the metric value to choose the best epoch if user has set the metric (``-x``). If not, xLearn uses the test_loss to choose the best epoch.

Lock-Free Learning
----------------------------------------

By default, xLearn performs Hogwild! lock-free learning, which takes advantages of multiple cores of 
modern CPU to accelerate training task. But lock-free training is non-deterministic. For example, if we 
run the following command multiple times, we may get different loss value at each epoch: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

   The 1st time: 0.449056
   The 2nd time: 0.449302
   The 3nd time: 0.449185

Users can set the number of thread for xLearn by using ``nthread`` parameter: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'nthread':4} 
            
   ffm_model.fit(param, "./model.out") 

xLearn will show the number of threads: ::

    [------------] xLearn uses 4 threads for training task.
    [ ACTION     ] Read Problem ...

Users can also disable lock-free training by using ``disableLockFree()`` API: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   ffm_model.disableLockFree()
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

In this time, our result are *deterministic*: ::

   The 1st time: 0.449172
   The 2nd time: 0.449172
   The 3nd time: 0.449172

The disadvantage of ``disableLockFree()`` is that it is much slower than lock-free training.

Instance-wise Normalization
----------------------------------------

For FM and FFM, xLearn uses *instance-wise normalizarion* by default. In some scenes like CTR prediction, 
this technique is very useful. But sometimes it hurts model performance. Users can disable instance-wise 
normalization by using ``disableNorm()`` API: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   ffm_model.disableNorm()
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

Note that if you use Instance-wise Normalization in training process, you also need to use the meachnism in prediction process.

Quiet Training
----------------------------------------

When using ``setQuiet()`` API, xLearn will not calculate any evaluation information during 
the training, and it just train the model quietly: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   ffm_model.setQuiet()
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

In this way, xLearn can accelerate its training speed significantly.

DMatrix Transition
----------------------------------------
Here is a simple Python demo to show that how to use xLearn python DMatrix API. You can checkout the 
demo data (``house_price_train.txt`` and ``house_price_test.txt``) from the path ``demo/regression/house_price``.

.. code-block:: python

    import xlearn as xl
    import numpy as np
    import pandas as pd

    # read file from file
    house_price_train = pd.read_csv("house_price_train.txt", header=None, sep="\t")
    house_price_test = pd.read_csv("house_price_test.txt", header=None, sep="\t")
    
    # get train X, y
    X_train = house_price_train[house_price_train.columns[1:]]
    y_train = house_price_train[0]

    # get test X, y
    X_test = house_price_test[house_price_test.columns[1:]]
    y_test = house_price_test[0]
    
    # DMatrix transition, if use field ,use must pass field map(an array) of features 
    xdm_train = xl.DMatrix(X_train, y_train)
    xdm_test = xl.DMatrix(X_test, y_test)

    # Training task
    fm_model = xl.create_fm()  # Use factorization machine
    # we use the same API for train from file
    # that is, you can also pass xl.DMatrix for this API now
    fm_model.setTrain(xdm_train)    # Training data
    fm_model.setValidate(xdm_test)  # Validation data
    
    # param:
    #  0. regression task
    #  1. learning rate: 0.2
    #  2. regular lambda: 0.002
    #  3. evaluation metric: mae
    param = {'task':'reg', 'lr':0.2, 
             'lambda':0.002, 'metric':'mae'}

    # Start to train
    # The trained model will be stored in model.out
    fm_model.fit(param, './model_dm.out')

    # Prediction task
    # we use the same API for test from file
    # that is, you can also pass xl.DMatrix for this API now
    fm_model.setTest(xdm_test)  # Test data

    # Start to predict
    # The output result will be stored in output.txt
    # if no result out path setted, we return res as numpy.ndarray
    res = fm_model.predict("./model_dm.out")

**Note:** Train from DMatrix is not support cross validation now, and we will add this feature soon later. 

Scikit-learn API for xLearn
----------------------------------------

xLearn can support scikit-learn-like api for users. Here is an example: ::

  import numpy as np
  import xlearn as xl
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split

  # Load dataset
  iris_data = load_iris()
  X = iris_data['data']
  y = (iris_data['target'] == 2)

  X_train,   \
  X_val,     \
  y_train,   \
  y_val = train_test_split(X, y, test_size=0.3, random_state=0)

  # param:
  #  0. binary classification
  #  1. model scale: 0.1
  #  2. epoch number: 10 (auto early-stop)
  #  3. learning rate: 0.1
  #  4. regular lambda: 1.0
  #  5. use sgd optimization method
  linear_model = xl.LRModel(task='binary', init=0.1, 
                            epoch=10, lr=0.1, 
                            reg_lambda=1.0, opt='sgd')

  # Start to train
  linear_model.fit(X_train, y_train, 
                   eval_set=[X_val, y_val], 
                   is_lock_free=False)

  # Generate predictions
  y_pred = linear_model.predict(X_val)

In this example, we use linear model to train a binary classifier. We can also 
create FM and FFM by using ``xl.FMModel()`` and ``xl.FFMModel()`` . Please see 
the details of these examples in (`Link`__)

.. __: https://github.com/aksnzhy/xlearn/tree/master/demo/classification/scikit_learn_demo
