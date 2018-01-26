xLearn API List
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This page gives the xLearn API List for the command line, Python package, and R package.

xLearn Command Line Usage
------------------------------

For Training: ::

    xlearn_train <train_file_path> [OPTIONS]

Options: ::

  -s <type> : Type of machine learning model (default 0)
     for classification task:
         0 -- linear model (GLM)
         1 -- factorization machines (FM)
         2 -- field-aware factorization machines (FFM)
     for regression task:
         3 -- linear model (GLM)
         4 -- factorization machines (FM)
         5 -- field-aware factorization machines (FFM)
                                                                           
  -x <metric>          :  The metric can be 'acc', 'prec', 'recall', 'f1', 'auc' for classification, and
                          'mae', 'mape', 'rmsd (rmse)' for regression. On defaurt, xLearn will not print
                          any evaluation metric information (only loss value).                                           
                                                                                                     
  -p <opt_method>      :  Choose the optimization method, including 'sgd', adagrad', and 'ftrl'. On default,
                          we use the 'adagrad' optimization.
                                                                                                
  -v <validate_file>   :  Path of the validation data file. This option will be empty by default,
                          and in this way, the xLearn will not perform validation process.
                                                                                             
  -m <model_file>      :  Path of the model dump file. On default, the model file name is 'train_file' + '.model'. 
                          If we set this value to 'none', the xLearn will not dump the model checkpoint after training.

  -t <txt_model_file>  :  Path of the txt model checkpoint file. On default, we do not set this option
                          and xLearn will not dump the txt model.
                                                                            
  -l <log_file>        :  Path of the log file. Using '/tmp/xlearn_log.*' by default.
                                                                                      
  -k <number_of_K>     :  Number of the latent factor used by fm and ffm tasks. Using 4 by default.
                          Note that, we will get the same model size when setting k to 1 and 4.
                          This is because we use SSE instruction and the memory need to be aligned.
                          So even you assign k = 1, we still fill some dummy zeros from k = 2 to 4.
                                                                                         
  -r <learning_rate>   :  Learning rate for optimization method. Using 0.2 by default.
                          xLearn can use adaptive gradient descent (AdaGrad) for optimization problem,
                          if you choose AdaGrad method, the learning rate will be changed adaptively.
                                                                                    
  -b <lambda_for_regu> :  Lambda for L2 regular. Using 0.00002 by default. We can disable the
                          regular term by setting this value to 0.0
                                                                     
  -u <model_scale>     :  Hyper parameter used for initialize model parameters.
                          Using 0.66 by default.
                                                                                 
  -e <epoch_number>    :  Number of epoch for training. Using 10 by default. Note that, xLearn will
                          perform early-stopping by default, so this value is just a upper bound.
                                                                                       
  -f <fold_number>     :  Number of folds for cross-validation. Using 5 by default.    

  -nthread <thread number> :  Number of thread for multi-thread training.
                                                                                     
  --disk               :  Open on-disk training for large-scale machine learning problems.
                                                                   
  --cv                 :  Open cross-validation in training tasks. If we use this option, xLearn
                          will ignore the validation file (-t). 
                                                                  
  --dis-lock-free      :  Disable lock-free training. Lock-free training can accelerate training but
                          the result is non-deterministic. Our suggestion is that you can open this flag
                          if the training data is big and sparse.
                                                                       
  --dis-es             :  Disable early-stopping in training. By default, xLearn will use early-stopping
                          in training tasks, except for training in cross-validation.
                                                                                         
  --no-norm            :  Disable instance-wise normalization. By default, xLearn will use
                          instance-wise normalization for both training and prediction.
                                                                 
  --quiet              :  Don't print any evaluation information during the training and
                          just train the model quietly.

  -alpha               :  Hyper parameters used by ftrl.
                                       
  -beta                :  Hyper parameters used by ftrl.
                                       
  -lambda_1            :  Hyper parameters used by ftrl.
                                       
  -lambda_2            :  Hyper parameters used by ftrl.

For Prediction: ::

    xlearn_predict <test_file> <model_file> [OPTIONS]

Options: ::

  -o <output_file>     :  Path of the output file. On default, this value will be set
                          to 'test_file' + '.out'
                                                      
  -l <log_file_path>   :  Path of the log file. Using '/tmp/xlearn_log' by default.  

  -nthread <thread number> :  Number of thread for multi-thread training.

  --sign                   :  Converting output to 0 and 1.

  --sigmoid                :  Converting output to 0~1 (problebility).

xLearn Python API
------------------------------

API List: ::

    import xlearn as xl      # Import xlearn package

    xl.hello()               # Say hello to user

    model = create_linear()  #  Create linear model.

    model = create_fm()      #  Create factorization machines.

    model = create_ffm()     #  Create field-aware factorizarion machines.

    model.show()             #  Show model information.

    model.fit(param, "model_path")   #  Train model.

    model.cv(param)    # Perform cross-validation.

    model.predict("model_path", "output_path")  # Perform prediction. 

    model.setTrain("data_path")      #  Set training data for xLearn.

    model.setValidate("data_path")   #  Set validation data for xLearn.

    model.setTest("data_path")       #  Set test data for xLearn.

    model.setQuiet()    #  Set xlearn to train model quietly.

    model.setOnDisk()   #  Set xlearn to use on-disk training.

    model.setSign()     # Convert prediction to 0 and 1.

    model.setSigmoid()  # Convert prediction to (0, 1).

    model.disableNorm() # Disable instance-wise normalization.

    model.disableLockFree()   # Disable lock-free training.

    model.disableEarlyStop()  # Disable early-stopping.

Parameter List: ::

    task     : {'binary',  # Binary classification
                'reg'}     # Regression

    metric   : {'acc', 'prec', 'recall', 'f1', 'auc',   # for classification
                'mae', 'mape', 'rmse', 'rmsd'}  # for regression

    lr       : float value  # learning rate

    lambda   : float value  # regular lambda

    k        : int value    # latent factor for fm and ffm

    init     : float value  # model initialize

    alpha    : float value  # hyper parameter for ftrl

    beta     : float value  # hyper parameter for ftrl

    lambda_1 : float value  # hyper parameter for ftrl

    lambda_2 : float value  # hyper parameter for ftrl

    epoch    : int vlaue    # number of epoch

    fold     : int value    # number of fold for cross-validation

    opt      : {'sgd', 'agagrad', 'ftrl'}  # optimization method

xLearn R API
------------------------------

xLearn R API page is coming soon.
