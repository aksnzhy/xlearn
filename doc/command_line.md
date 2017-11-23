## Command Line Usage

Here we use a real example to demonstrate how to use xLearn by command line. 
Make sure that you have already installed the executable file (`xlearn_train` and
`xlearn_predict`) successfully. See this page [install.md][1] for installation.

The training data - `small_train.txt` and testing data - `small_test.txt` (in the root directory of your xlearn build package) is a portion of the whole data of criteo ctr prediction challenge in [kaggle][2].

### Quck start

We can use `xlearn_train` to train our model. 

Usage: xlearn_teain train_data [OPTIONS]

  [1]: install.md
  [2]: https://www.kaggle.com/c/criteo-display-ad-challenge