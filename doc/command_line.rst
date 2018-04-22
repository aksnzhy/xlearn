xLearn Command Line Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you built xLearn from source code successfully, you will get two executable files 
``xlearn_train`` and ``xlearn_predict`` in your ``build`` directory. Now you can use these 
two executable files to perform training and prediction tasks.

Quick Start
----------------------------------------

Make sure that you are in the ``build`` directory of xLearn, and you can find the demo data 
``small_test.txt`` and ``small_train.txt`` in this directory. Now type the following 
command to train a model: ::

    ./xlearn_train ./small_train.txt

Here, we show a portion of the xLearn's output. Note that the loss value shown in your machine 
could be different. ::

    Epoch      Train log_loss     Time cost (sec)
        1            0.567514                0.00
        2            0.516861                0.00
        3            0.489884                0.00
        4            0.469971                0.00
        5            0.452699                0.00
        6            0.437590                0.00
        7            0.425759                0.00
        8            0.415190                0.00
        9            0.405954                0.00
       10            0.396313                0.00

By default, xLearn will use the logistic regression (LR) to train our model within 10 epoch.

We can see that a new file called ``small_train.txt.model`` has been generated in the current directory. 
This file stores the trained model checkpoint, and we can use this model file to make prediction in 
the future ::

    ./xlearn_predict ./small_test.txt ./small_train.txt.model

After that, we can get a new file called ``small_test.txt.out`` in the current directory. This is output 
prediction. Here we show the first five lines of this output by using the following command ::
    
    head -n 5 ./small_test.txt.out

    -1.9872
    -0.0707959
    -0.456214
    -0.170811
    -1.28986

These lines of data are the prediction score calculated for examples in the test set. The 
negative data represents the negative example and positive data represents the positive example. 
In xLearn, you can convert the score to (0-1) by using ``--sigmoid`` option, or you can convert 
your result to binary result (0 and 1) by using ``--sign`` option ::

    ./xlearn_predict ./small_test.txt ./small_train.txt.model --sigmoid
    head -n 5 ./small_test.txt.out

    0.120553
    0.482308
    0.387884
    0.457401
    0.215877

    ./xlearn_predict ./small_test.txt ./small_train.txt.model --sign
    head -n 5 ./small_test.txt.out

    0
    0
    0
    0
    0

Users may want to generate different model files, so you can set the name of the model 
checkpoint file by using ``-m`` option. By default, the name of the model file equals to
``training_data_name`` + ``.model`` ::

  ./xlearn_train ./small_train.txt -m new_model

Also, users can save the model in txt format by using ``-t`` option. For example: ::

  ./xlearn_train ./small_train.txt -t model.txt

After that, we get a new file called ``model.txt``, which stores the trained model in txt format. ::

  head -n 5 ./model.txt

  -0.688182
  0.458082
  0
  0
  0

For the linear and bias term, we store each parameter in each line. For FM and FFM, we store one 
vector of the latent factor in each line.

Users can also set ``-o`` option to specify the output file. For example: ::

  ./xlearn_predict ./small_test.txt ./small_train.txt.model -o output.txt  
  head -n 5 ./output.txt

  -2.01192
  -0.0657416
  -0.456185
  -0.170979
  -1.28849

By default, the name of the output file equals to ``test_data_name`` + ``.out`` .

Choose Machine Learning Algorithm
----------------------------------------

For now, xLearn can support three different machine learning algorithms, including LR, FM and FFM.
Users can choose different machine learning algorithms by using ``-s`` option: ::

  -s <type> : Type of machine learning model (default 0)
     for classification task:
         0 -- linear model (GLM)
         1 -- factorization machines (FM)
         2 -- field-aware factorization machines (FFM)
     for regression task:
         3 -- linear model (GLM)
         4 -- factorization machines (FM)
         5 -- field-aware factorization machines (FFM)

For LR and FM, the input data format can be ``CSV`` or ``libsvm``. For FFM, the 
input data should be the ``libffm`` format. ::

  libsvm format:

     label index_1:value_1 index_2:value_2 ... index_n:value_n

  CSV format:

     label value_1 value_2 .. value_n

Note that, if the csv file doesn't contain the label y, the user should add a 
placeholder to the dataset by themselves (Also in test data). Otherwise, the parser 
will treat the first element as the label y. 

  libffm format:

     label field_1:index_1:value_1 field_2:index_2:value_2 ...

Users can also give a ``libffm`` file to LR and FM. At that time, xLearn will treat this data 
as ``libsvm`` format. The following command shows how to use different
machine learning algorithms to solve the binary classification problem:  ::

./xlearn_train ./small_train.txt -s 0  # Linear model
./xlearn_train ./small_train.txt -s 1  # Factorization machine (FM)
./xlearn_train ./small_train.txt -s 2  # Field-awre factorization machine (FFM)

Set Validation Dataset
----------------------------------------

A validation dataset is used to tune the hyperparameters of a machine learning model. 
In xLearn, users can use ``-v`` option to set the validation dataset. For example: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt    

A portion of xLearn's output: ::

    Epoch      Train log_loss       Test log_loss     Time cost (sec)
        1            0.575049            0.530560                0.00
        2            0.517496            0.537741                0.00
        3            0.488428            0.527205                0.00
        4            0.469010            0.538175                0.00
        5            0.452817            0.537245                0.00
        6            0.438929            0.536588                0.00
        7            0.423491            0.532349                0.00
        8            0.416492            0.541107                0.00
        9            0.404554            0.546218                0.00

Here we can see that the training loss continuously goes down. But the validation loss (test loss) goes down 
first, and then goes up. This is because our model has already overfitted current training dataset. By default, 
xLearn will calculate the validation loss in each epoch, while users can also set different evaluation
metrics by using ``-x`` option. For classification problems, the metric can be : ``acc`` (accuracy), ``prec`` 
(precision), ``f1`` (f1 score), ``auc`` (AUC score). For example: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt -x acc
    ./xlearn_train ./small_train.txt -v ./small_test.txt -x prec
    ./xlearn_train ./small_train.txt -v ./small_test.txt -x f1
    ./xlearn_train ./small_train.txt -v ./small_test.txt -x auc

For regression problems, the metric can be ``mae``, ``mape``, and ``rmsd`` (rmse). For example: ::

    cd demo/house_price/
    ../../xlearn_train ./house_price_train.txt -s 3 -x rmse --cv
    ../../xlearn_train ./house_price_train.txt -s 3 -x rmsd --cv

Cross-Validation
----------------------------------------

Cross-validation, sometimes called rotation estimation, is a model validation technique for assessing 
how the results of a statistical analysis will generalize to an independent dataset. In xLearn, users 
can use the ``--cv`` option to use this technique. For example: ::

    cd build
    ./xlearn_train ./small_train.txt --cv

On default, xLearn uses 5-folds cross validation, and users can set the number of fold by using 
``-f`` option: ::
    
    ./xlearn_train ./small_train.txt -f 3 --cv

Here we set the number of folds to 3. The xLearn will calculate the average validation loss at 
the end of its output message. ::

    [------------] Average log_loss: 0.549417
    [ ACTION     ] Finish Cross-Validation
    [ ACTION     ] Clear the xLearn environment ...
    [------------] Total time cost: 0.03 (sec)

Choose Optimization Method
----------------------------------------
 
In xLearn, users can choose different optimization methods by using ``-p`` option. For now, users can 
choose ``sgd``, ``adagrad``, and ``ftrl`` method. By default, xLearn uses the ``adagrad`` method. 
For example: ::

    ./xlearn_train ./small_train.txt -p sgd
    ./xlearn_train ./small_train.txt -p adagrad
    ./xlearn_train ./small_train.txt -p ftrl

Compared to traditional sgd method, adagrad adapts the learning rate to the parameters, performing larger 
updates for infrequent and smaller updates for frequent parameters. For this reason, it is well-suited for 
dealing with sparse data. In addition,  sgd is more sensitive to the learning rate compared with adagrad.

FTRL (Follow-the-Regularized-Leader) is also a famous method that has been widely used in the large-scale 
sparse problem. To use FTRL, users need to tune more hyperparameters compared with sgd and adagard. 

Hyperparameter Tuning
----------------------------------------

In machine learning, a *hyperparameter* is a parameter whose value is set before the learning process begins. 
By contrast, the value of other parameters is derived via training. Hyperparameter tuning is the problem of 
choosing a set of optimal hyperparameters for a learning algorithm.

First, the ``learning rate`` is one of the most important hyperparameters used in machine learning. 
By default, this value is set to ``0.2``, and we can tune this value by using ``-r`` option: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.5
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.01


We can also use the ``-b`` option to perform regularization. By default, xLearn uses ``L2`` regularization, and 
the *regular_lambda* has been set to ``0.00002``. ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1 -b 0.001
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1 -b 0.002
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1 -b 0.01


For the FTRL method, we also need to tune another four hyperparameters, including ``-alpha``, ``-beta``, 
``-lambda_1``, and ``-lambda_2``. For example: ::

    ./xlearn_train ./small_train.txt -p ftrl -alpha 0.002 -beta 0.8 -lambda_1 0.001 -lambda_2 1.0

For FM and FFM, users also need to set the size of *latent factor* by using ``-k`` option. By default, xLearn 
uses ``4`` for this value. ::

    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 2
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 4
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 5
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 8

xLearn uses *SSE* instruction to accelerate vector operation, and hence the time cost for ``k=2`` and ``k=4`` are the same.

For FM and FFM, users can also set the hyperparameter ``-u`` for model initialization. By default, this value is set to 0.66. ::

    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -u 0.80
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -u 0.40
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -u 0.10

Set Epoch Number and Early-Stopping
----------------------------------------

For machine learning, one epoch consists of one full training cycle on the training set. 
In xLearn, users can set the number of epoch for training by using ``-e`` option. ::

    ./xlearn_train ./small_train.txt -e 3
    ./xlearn_train ./small_train.txt -e 5
    ./xlearn_train ./small_train.txt -e 10   

If you set the validation data, xLearn will perform early-stopping by default. For example: ::
  
    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -e 10

Here, we set epoch number to ``10``, but xLearn stopped at epoch 7 because we get the best model 
at that epoch (you may get different a stopping number on your machine) ::

  [ ACTION     ] Early-stopping at epoch 7
  [ ACTION     ] Start to save model ...

User can set window size for early stopping by using ``-sw`` option. ::

    ./xlearn_train ./small_train.txt -e 10 -sw 3

Users can disable early-stopping by using ``--dis-es`` option ::

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -e 10 --dis-es

At this time, xLearn performed 10 epoch for training.

Lock-Free Training
----------------------------------------

By default, xLearn performs *Hogwild! lock-free* training, which takes advantages of multiple cores to accelerate
training task. But lock-free training is *non-deterministic*. For example, if we run the following command multiple 
times, we may get different loss value at each epoch. ::

   ./xlearn_train ./small_train.txt 

   The 1st time: 0.396352
   The 2nd time: 0.396119
   The 3nd time: 0.396187
   ...

Users can set the number of thread for xLearn by using ``-nthread`` option: ::

   ./xlearn_train ./small_train.txt -nthread 2

If you don't set this option, xLearn uses all of the CPU cores by default.

Users can disable lock-free training by using ``--dis-lock-free`` ::

  ./xlearn_train ./small_train.txt --dis-lock-free

In thie time, our result are *determinnistic*. ::

   The 1st time: 0.396372
   The 2nd time: 0.396372
   The 3nd time: 0.396372

The disadvantage of ``--dis-lock-free`` is that it is much slower than lock-free training. 

Instance-wise Normalization
----------------------------------------

For FM and FFM, xLearn uses *instance-wise normalizarion* by default. In some scenes like CTR prediction, this technique is very
useful. But sometimes it hurts model performance. Users can disable instance-wise normalization by using ``--no-norm`` option ::

  ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt --no-norm

Note that we usually use ``--no-norm`` in regression tasks.

Quiet Training
----------------------------------------

When using ``--quiet`` option, xLearn will not calculate any evaluation information during the training, and 
it just train the model quietly ::

  ./xlearn_train ./small_train.txt --quiet

In this way, xLearn can accelerate its training speed.


 .. toctree::
   :hidden: