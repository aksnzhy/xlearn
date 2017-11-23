## Command Line Usage

Here we use a real example to demonstrate how to use xLearn by command line. 
Make sure that you have already installed the executable file (`xlearn_train` and
`xlearn_predict`) successfully. See this page [install.md][1] for installation.

The training data - `small_train.txt` and testing data - `small_test.txt` (in the root directory of your xlearn build package) is a portion of the whole data of criteo ctr prediction challenge in [kaggle][2].

#### Quck start

We can use `xlearn_train` to train our model.  Usage:  

    ./xlearn_train train_data [OPTIONS]

For example: 

    ./xlearn_train ./small_train.txt -s 2

#### Choose machine learning model

We use `-s` to specify the machine learning model we want. Here we use ffm.

    -s : Type of machine learning model (default 0)
         for classification task:
             0 -- linear model (GLM)
             1 -- factorization machines (FM)
             2 -- field-aware factorization machines (FFM)
         for regression task:
             3 -- linear model (GLM)
             4 -- factorization machines (FM)
             5 -- field-aware factorization machines (FFM)

#### Set validation

We can also use `-v` to specify the validation data, for example:

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt

#### Early-stopping
    
On default, xlearn adopts `early-stopping` when we set the validation data. You can disable early-stopping by using option `--dis-es`, for example:

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt --dis-es

#### Evaluation metric

Also, we can set specify evaluation metric by using `-x` option. For example:

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -x auc

where xlearn will print the AUC value. xLearn support a set evaluation metric, including:

    -x: 
     acc
     prec
     recall
     f1
     auc
     mae
     mape
     rmsd
     rmse

#### Learning rate

You can set the learning rate by using `-r` option. On default, the learning rate is 0.2.

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -r 0.1

#### Regular lambda

You can set the regular lambda by using `-b` option. On default, this option is set to 0.0002.

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -b 0.001

#### Number of latent factor (K)

For fm and ffm tasks, users can set the size of latent factor. On default this option is set to 4.

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -k 8


#### Number of epoch

We set the number of epoch by using `-e` option, for example:

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -e 15

#### Corss-validation

We can use cross-validation in xLearn by setting the `--cv` option, for example:

    ./xlearn_train ./small_train.txt -s 2 --cv

On default, we use 4-fold cross-validation. We can also the fold number by using `-f` option. For example:

    ./xlearn_train ./small_train.txt -s 2 -f 3 --cv

#### Lock free learning

On default, xLearn uses lock-free training (like Hogwild!). So when run our problem multiple times, the result could be non-deterministic. If you want to get a deterministic result, we can disable lock-free training by setting `--dis-lock-free` option.

#### Normalization 

On default, xLearn uses instance-wise normalization. You can disable this normalization by setting option `--dis-norm`

#### Quite training

We can set `--quiet` option to disable any evaluation output.

#### Model scale

Model scale is hyper-parameter used for initialize model parameters. The default value is 0.66 and we can change this value by using `-u` option.

#### Model output

You can also specify your output file for trainned model by using `-m`. If you don't set this option, xlearn we dump the model to the file `small_train.txt.model` by default.

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -m ./model.out

#### Set log

You can specify the path of your log path by using `-l` option. On default, xlearn will save the log information in `/tmp/`

#### Help menu

You can type ./xlearn_train, and then xlearn will print the complete command line usage formation:

    USAGE:
         xlearn_train <train_file_path> [OPTIONS]
    
     e.g.,  xlearn_train train_data.txt -s 0 -v validate_data.txt -r 0.1
    
    OPTIONS:
      -s <type> : Type of machine learning model (default 0)
         for classification task:
             0 -- linear model (GLM)
             1 -- factorization machines (FM)
             2 -- field-aware factorization machines (FFM)
         for regression task:
             3 -- linear model (GLM)
             4 -- factorization machines (FM)
             5 -- field-aware factorization machines (FFM)
    
      -x <metric>          :  The metric can be 'acc', 'prec', 'recall', 'f1' (classification), and 'mae',
                              'mape', 'rmsd (rmse)' (regression). xLearn uses the Accuracy (acc) by default.
                              If we set this option to 'none', xLearn will not print any metric information.
    
      -v <validate_file>   :  Path of the validation data file. This option will be empty by default,
                              and in this way, the xLearn will not perform validation.
    
      -m <model_file>      :  Path of the model checkpoint file. On default, the model file name will be.
                              set to 'train_file' + '.model'. If we set this value to 'none', the xLearn will
                              not dump the model checkpoint after training.
    
      -l <log_file>        :  Path of the log file. Using '/tmp/xlearn_log/' by default.
    
      -k <number_of_K>     :  Number of the latent factor used by fm and ffm tasks. Using 4 by default.
                              Note that, we will get the same model size when setting k to 1 and 4.
                              This is because we use SSE instruction and the memory need to be aligned.
                              So even you assign k = 1, we still fill some dummy zeros from k = 2 to 4.
    
      -r <learning_rate>   :  Learning rate for stochastic gradient descent. Using 0.2 by default.
                              xLearn uses adaptive gradient descent (AdaGrad) for optimization problem,
                              and the learning rate will be changed adaptively.
    
      -b <lambda_for_regu> :  Lambda for L2 regular. Using 0.00002 by default. We can disable the
                              regular term by setting this value to 0.0
    
      -u <model_scale>     :  Hyper parameter used for initialize model parameters.
                              Using 0.66 by default.
    
      -e <epoch_number>    :  Number of epoch for training. Using 10 by default. Note that, xLearn will
                              perform early-stopping by default, so this value is just a upper bound.
    
      -f <fold_number>     :  Number of folds for cross-validation. Using 5 by default.
    
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



  [1]: install.md
  [2]: https://www.kaggle.com/c/criteo-display-ad-challenge