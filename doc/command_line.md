## Command Line Usage

Here we use a real example to demonstrate how to use xLearn by command line. 
Make sure that you have already installed the executable file (`xlearn_train` and
`xlearn_predict`) successfully. See this page [install.md][1] for installation.

The training data - `small_train.txt` and testing data - `small_test.txt` (in the root directory of your xlearn build package) is a portion of the whole data of criteo ctr prediction challenge in [kaggle][2].

### Quck start

We can use `xlearn_train` to train our model.  Usage:  

    ./xlearn_train train_data [OPTIONS]

For example: 

    ./xlearn_train ./small_train.txt -s 2

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

We can also use `-v` to specify the validation data, for example:

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt
    
On default, xlearn adopts `early-stopping` when we set the validation data. You can disable early-stopping by using option `--dis-es`

Also, we can set specify evaluation metric by using `-x` option. For example:

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -x auc

, where xlearn will print the AUC value. xLearn support a set evaluation metric, including:

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

  [1]: install.md
  [2]: https://www.kaggle.com/c/criteo-display-ad-challenge