xLearn Python Package Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^

xLearn supports very easy-to-use Python API for users. Once you install the 
xLearn Python package successfully, you can try it. Type ``python`` in your
shell and type the following Python code to check your installation: ::

    import xlearn as xl
    xl.hello()

If you install xLearn Python package successfully, you will see ::

  -------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.30 Version --
  -------------------------------------------------------------------------

Quick Start
----------------------------------------

Here is a simple Python demo to demonstrate how to use xLearn. You can checkout the demo data 
(``small_train.txt`` and ``small_test.txt``) from the path ``demo/classification/criteo_ctr``.

.. code-block:: python

   import xlearn as xl

   # Training task
   ffm_model = xl.create_ffm()  # Use field-aware factorization machine
   ffm_model.setTrain("./small_train.txt")   # Training data
   ffm_model.setValidate("./small_test.txt")  # Validation data

   # param:
   #  0. binary classification
   #  1. learning rate : 0.2
   #  2. regular lambda : 0.002
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002}
            
   # Train model
   ffm_model.fit(param, "./model.out")  

A portion of the xLearn's output ::
  
  Start to train ...
  Epoch      Train log_loss       Test log_loss     Time cost (sec)
      1            0.593750            0.535847                0.00
      2            0.539226            0.543829                0.00
      3            0.520034            0.531732                0.00
      4            0.505186            0.537418                0.00
      5            0.494089            0.533448                0.00
      6            0.483678            0.534629                0.00
      7            0.470848            0.528086                0.00
      8            0.466330            0.533253                0.00
      9            0.456660            0.535635                0.00
  Early-stopping at epoch 7
  Start to save model ...

In this example, xLearn uses *feild-ware factorization machines* (ffm) to train our model for 
solving a binary classification task. If you want train a model for regression task. 
You can reset the ``task`` parameter to ``reg``. ::

    param = {'task':'reg', 'lr':0.2, 'lambda':0.002} 

We can see that a new file called ``model.out`` has been generated in the current directory. 
This file stores the trained model checkpoint, and we can use this model file to make prediction 
in the future: ::

    ffm_model.setTest("./small_test.txt")
    ffm_model.predict("./model.out", "./output.txt")      

After we run this Python code, we can get a new file called ``output.txt`` in current directory. 
This is output prediction. Here we show the first five lines of this output by using the following command ::

    head -n 5 ./output.txt

    -1.66107
    -0.616408
    -0.815918
    -0.608931
    -1.30794

These lines of data are the prediction score calculated for examples in the test set. The negative data 
represents the negative example and positive data represents the positive example. In xLearn, you can convert 
the score to (0-1) by using ``setSigmoid()`` option: ::

   ffm_model.setTest("./small_test.txt")  
   ffm_model.setSigmoid()
   ffm_model.predict("./model.out", "./output.txt")      

and then we can get the result ::

   head -n 5 ./output.txt

   0.158613
   0.354297
   0.310193
   0.357449
   0.220061

We can also convert the score to binary result ``(0 and 1)`` by using ``setSign()`` API ::

   # Prediction task
   ffm_model.setTest("./small_test.txt")  
   ffm_model.setSign()
   ffm_model.predict("./model.out", "./output.txt")

and then we can get the result ::

   head -n 5 ./output.txt

   0
   0
   0
   0
   0

Also, users can save the model in txt format by using ``setTXTModel()`` API. For example: ::

    ffm_model.setTXTModel("./model.txt")

After that, we get a new file called ``model.txt``, which stores the trained model in txt format.::

  head -n 5 ./model.txt

  -0.688182
  0.458082
  0
  0
  0

For the linear and bias term, we store each parameter in each line. For FM and FFM, we store one 
vector of the latent factor in each line.

Choose Machine Learning Algorithm
----------------------------------------

For now, xLearn can support three different machine learning algorithms, including LR, FM and FFM. 
Users can choose different machine learning algorithms by using ``create_xxx()`` API: ::
   
    import xlearn as xl

    ffm_model = xl.create_ffm()
    fm_model = xl.create_fm()
    lr_model = xl.create_lr()


For LR and FM, the input data format can be ``CSV`` or ``libsvm``. For FFM, the input data should 
be the ``libffm`` format. ::

  libsvm format:

    label index_1:value_1 index_2:value_2 ... index_n:value_n

  CSV format:

    value_1 value_2 .. value_n label

  libffm format:

    label field_1:index_1:value_1 field_2:index_2:value_2 ...

Users can also give a libffm file to LR and FM. At that time, xLearn will treat this data as libsvm format. 

Set Validation Dataset
----------------------------------------

A validation dataset is used to tune the hyperparameters of a machine learning model. In xLearn, users can 
use ``setValdiate()`` API to set the validation dataset. For example: ::

   import xlearn as xl

   # Training task
   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")
   ffm_model.setValidate("./small_test.txt")  
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

A portion of xLearn's output: ::

   Epoch      Train log_loss       Test log_loss     Time cost (sec)
       1            0.598814            0.536327                0.00
       2            0.539872            0.542924                0.00
       3            0.521035            0.531595                0.00
       4            0.505414            0.536246                0.00
       5            0.492150            0.532070                0.00
       6            0.482229            0.536482                0.00
       7            0.470457            0.528871                0.00
       8            0.464445            0.534550                0.00
       9            0.456061            0.537320                0.00

Here we can see that the training loss continuously goes down. But the validation loss (test loss) 
goes down first, and then goes up. This is because our model has already overfitted current training 
dataset. By default, xLearn will calculate the validation loss in each epoch, while users can also 
set different evaluation metrics by using ``metric`` parameter. For classification problems, the metric can be : 
``acc`` (accuracy), ``prec`` (precision), ``f1`` (f1 score), and ``auc`` (AUC score). 
For example: ::

   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'acc'}
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'prec'}
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'f1'}
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'auc'}           

For regression problems, the metric can be ``mae``, ``mape``, and ``rmsd`` (rmse). 
For example: ::

   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'rmse'}
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'mae'}    
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'mape'}  

Cross-Validation
----------------------------------------

Cross-validation, sometimes called rotation estimation, is a model validation technique for assessing 
how the results of a statistical analysis will generalize to an independent dataset. In xLearn, users 
can use the ``cv()`` API to use this technique. For example: ::

    import xlearn as xl

    # Training task
    ffm_model = xl.create_ffm()
    ffm_model.setTrain("./small_train.txt")  
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
    ffm_model.cv(param)

On default, xLearn uses 5-folds cross validation, and users can set the number of fold by 
using the ``fold`` parameter: ::

    import xlearn as xl

    # Training task
    ffm_model = xl.create_ffm()
    ffm_model.setTrain("./small_train.txt")  
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'fold':3} 
            
    ffm_model.cv(param)     

Here we set the number of folds to 3. The xLearn will calculate the average validation loss at the 
end of its output message. ::

   [------------] Average log_loss: 0.547632
   [ ACTION     ] Finish Cross-Validation
   [ ACTION     ] Clear the xLearn environment ...
   [------------] Total time cost: 0.05 (sec)

Choose Optimization Method
----------------------------------------

In xLearn, users can choose different optimization methods by using ``opt`` parameter. 
For now, users can choose ``sgd``, ``adagrad``, and ``ftrl`` method. By default, xLearn uses the ``adagrad`` method. 
For example: ::

   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'opt':'sgd'} 
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'opt':'adagrad'} 
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'opt':'ftrl'} 

Compared to traditional sgd method, adagrad adapts the learning rate to the parameters, performing larger updates 
for infrequent and smaller updates for frequent parameters. For this reason, it is well-suited for dealing with 
sparse data. In addition, sgd is more sensitive to the learning rate compared with adagrad.

FTRL (Follow-the-Regularized-Leader) is also a famous method that has been widely used in the large-scale sparse 
problem. To use FTRL, users need to tune more hyperparameters compared with sgd and adagard.

Hyperparameter Tuning
----------------------------------------

In machine learning, a *hyperparameter* is a parameter whose value is set before the learning process begins. 
By contrast, the value of other parameters is derived via training. Hyperparameter tuning is the problem of choosing 
a set of optimal hyperparameters for a learning algorithm.

First, the ``learning rate`` is one of the most important hyperparameters used in machine learning. By default, 
this value is set to 0.2, and we can tune this value by using ``lr`` parameter: ::

    param = {'task':'binary', 'lr':0.2} 
    param = {'task':'binary', 'lr':0.5}
    param = {'task':'binary', 'lr':0.01}

We can also use the ``lambda`` parameter to perform regularization. By default, xLearn uses L2 regularization, and 
the *regular_lambda* has been set to ``0.00002``. ::

    param = {'task':'binary', 'lr':0.2, 'lambda':0.01}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.02} 
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 

For the FTRL method, we also need to tune another four hyperparameters, 
including ``alpha``, ``beta``, ``lambda_1``, and ``lambda_2``. For example: ::

    param = {'alpha':0.002, 'beta':0.8, 'lambda_1':0.001, 'lambda_2': 1.0}    

For FM and FFM, users also need to set the size of latent factor by using ``k`` parameter. By default, 
xLearn uses ``4`` for this value. ::

    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':2}    
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':4}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':5}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':8}

xLearn uses *SSE* instruction to accelerate vector operation, and hence the time cost 
for ``k=2`` and ``k=4`` are the same.     

For FM and FFM, users can also set the parameter ``init`` for model initialization. 
By default, this value is set to ``0.66``.

    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'init':0.5}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'init':0.8}
  
Set Epoch Number and Early-Stopping
----------------------------------------

For machine learning, one epoch consists of one full training cycle on the training set. 
In xLearn, users can set the number of epoch for training by using ``epoch`` option. ::

    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'epoch':3}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'epoch':5}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'epoch':10}

If you set the validation data, xLearn will perform early-stopping by default. For example: ::

   import xlearn as xl

   # Training task
   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")
   ffm_model.setValidate("./small_test.txt")
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'epoch':10} 
            
   ffm_model.fit(param, "./model.out") 

Here, we set epoch number to ``10``, but xLearn stopped at epoch ``7`` because we get the best model 
at that epoch (you may get different a stopping number on your machine) ::

    Early-stopping at epoch 7
    Start to save model ...

Users can set stop window for early-stopping by using ``stop_window`` parameter ::

    param = {'task':'binary', 
             'lr':0.2, 
             'lambda':0.002, 
             'epoch':10,
             'stop_window':3} 
            
    ffm_model.fit(param, "./model.out") 

Users can disable early-stopping by using ``disableEarlyStop()`` API: ::

   import xlearn as xl

   # Training task
   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")
   ffm_model.setValidate("./small_test.txt")
   ffm_model.disableEarlyStop();
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'epoch':10} 
            
   ffm_model.fit(param, "./model.out") 

At this time, xLearn performed 10 epoch for training.

Lock-Free Training
----------------------------------------

By default, xLearn performs *Hogwild! lock-free* training, which takes advantages of multiple cores 
to accelerate training task. But lock-free training is *non-deterministic*. For example, if we run the 
following Python code multiple times, we may get different loss value at each epoch. ::

   import xlearn as xl

   # Training task
   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

   The 1st time: 0.449056
   The 2nd time: 0.449302
   The 3nd time: 0.449185

Users can disable lock-free training by using ``disableLockFree()`` API. ::

   import xlearn as xl

   # Training task
   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   ffm_model.disableLockFree()
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

In this time, our result are *deterministic*. ::

   The 1st time: 0.449172
   The 2nd time: 0.449172
   The 3nd time: 0.449172

The disadvantage of ``disableLockFree()`` is that it is much slower than lock-free training.

Instance-wise Normalization
----------------------------------------

For FM and FFM, xLearn uses instance-wise normalizarion by default. In some scenes like CTR prediction, 
this technique is very useful. But sometimes it hurts model performance. Users can disable *instance-wise normalization* 
by using ``disableNorm()`` API. ::

   import xlearn as xl

   # Training task
   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   ffm_model.disableNorm()
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

Note that we usually use ``disableNorm`` in regression tasks.

Quiet Training
----------------------------------------

When using ``setQuiet()`` API, xLearn will not calculate any evaluation information during the training, and it just train the model quietly ::

   import xlearn as xl

   # Training task
   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   ffm_model.setQuiet()
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

In this way, xLearn can accelerate its training speed.

Scikit-learn api for xLearn
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
create FM and FFM by using ``xl.FMModel()`` and ``xl.FMModel()`` . Please see 
the details of these examples in (`Link`__)

.. __: https://github.com/aksnzhy/xlearn/tree/master/demo/classification/scikit_learn_demo

 .. toctree::
   :hidden: