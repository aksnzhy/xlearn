//------------------------------------------------------------------------------
// Copyright (c) 2016 by contributors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

/*
Author: Chao Ma (mctt90@gmail.com)
This file is the implementation of the Checker class.
*/

#include <string>
#include <cstdlib>
#include <algorithm>
#include <cstdio>

#include "src/base/common.h"
#include "src/solver/checker.h"
#include "src/base/levenshtein_distance.h"
#include "src/base/file_util.h"

namespace xLearn {

// Option help menu
std::string Checker::option_help() const {
  if (is_train_) {
    return std::string(
"----------------------------------------  Training task  -------------------------------------\n"
"USAGE: \n"
"     xlearn_train [ train_file_path ] [ OPTIONS ] \n"
"                                                   \n"
"OPTIONS: \n"
"  -s <type> : Type of machine learning model (default 0) \n"
"     for classification task \n"
"         0 -- logistic regression (LR) \n"
"         1 -- linear support vectors machine (linear SVM) \n"
"         2 -- factorization machines (FM) \n"
"         3 -- field-aware factorization machines (FFM) \n"
"     for regression task \n"
"         4 -- linear regression (LR) \n"
"         5 -- factorization machines (FM) \n"
"         6 -- field-aware factorization machines (FFM) \n"
"                                                                            \n"
"  -x <metric>          :  The metric can be 'acc', 'prec', 'recall', 'f1', 'auc' (classification), \n"
"                          and 'mae', 'mape' (regression). Using 'acc' - Accuracy by default. \n "
"                          If we set this option to 'none', xlearn will not print any metric info. \n"
"                                                                                              \n"
"  -t <test_file_path>  :  Path of the test data file. This option will be empty by default, \n"
"                          and in this way, the xLearn will not perform validation. \n"
"                                                                                              \n"
"  -m <model_file_path> :  Path of the model checkpoint file. Using './xlearn_model' by default. \n"
"                          If we set this value to 'none', the xLearn will not dump the model \n"
"                          checkpoint after the training. \n"
"                                                              \n"
"  -l <log_file_path>   :  Path of the log file. Using '/tmp/xlearn_log/' by default. \n"
"                                                                                  \n"
"  -k <number_of_K>     :  Number of the latent factor for fm and ffm tasks. Using 4 by default. \n"
"                          Note that, because we use SSE, the memory should be aligned. \n"
"                                                                                         \n"
"  -r <learning_rate>   :  Learning rate for gradient descent. Using 0.2 by default. \n"
"                                                                                     \n"
"  -b <lambda_for_regu> :  Lambda for regular. Using 0.00002 by default. We can disable the \n"
"                          regular term by setting this value to 0.0 \n"
"                                                                   \n"
"  -u <model_scale>     :  Hyper param used for initialize model parameters. Using 0.66 by default. \n"
"                                                                             \n"
"  -e <epoch_number>    :  Number of epoch for training. Using 10 by default. \n"
"                                                                              \n"
"  -f <fold_number>     :  Number of folds for cross-validation. Using 5 by default. \n"
"                                                                                   \n"
"  --disk               :  Open on-disk training for limited memory. \n"
"                                                                    \n"
"  --cv                 :  Open cross-validation in training tasks. If we use this option, xLearn \n"
"                          will ignore the validation file (-t).  \n"
"                                                                   \n"
"  --es                 :  Open early-stopping in training. \n"
"                                                           \n"
"  --no-norm            :  Disable instance-wise normalization.  \n"
"                                                              \n"
"  --quiet              :  Don't print any evaluation information during the training and \n"
"                          just train the model quietly. \n"
"----------------------------------------------------------------------------------------------\n"
    );
  } else {
    return std::string(
"-------------------------------------- Prediction task ---------------------------------------\n"
"USAGE: \n"
"     xlearn_predict [ predict_file_path ] [ options ] \n"
"                                                     \n"
"OPTIONS: \n"
"  -m <model_file_path>  :  Path of the trained model file. \n"
"                           Using './xlearn_model' by default. \n"
"                                                               \n"
"  -o <output_file_path> :  Path of the output file \n"
"                           Using './xlearn_out' by default. \n"
"                                                             \n"
"  -l <log_file_path>    :  Path of the log file. Using '/tmp/xlearn_log' by default. \n"
"----------------------------------------------------------------------------------------------\n"
    );
  }
}

// Initialize Checker
void Checker::Initialize(bool is_train, int argc, char* argv[]) {
  is_train_ = is_train;
  if (is_train_) {
    menu_.push_back(std::string("-s"));
    menu_.push_back(std::string("-x"));
    menu_.push_back(std::string("-t"));
    menu_.push_back(std::string("-m"));
    menu_.push_back(std::string("-l"));
    menu_.push_back(std::string("-k"));
    menu_.push_back(std::string("-r"));
    menu_.push_back(std::string("-b"));
    menu_.push_back(std::string("-u"));
    menu_.push_back(std::string("-e"));
    menu_.push_back(std::string("-f"));
    menu_.push_back(std::string("--disk"));
    menu_.push_back(std::string("--cv"));
    menu_.push_back(std::string("--es"));
    menu_.push_back(std::string("--no-norm"));
    menu_.push_back(std::string("--quiet"));
  } else {  // for Predict
    menu_.push_back(std::string("-m"));
    menu_.push_back(std::string("-o"));
    menu_.push_back(std::string("-l"));
  }
  // Get the user input
  for (int i = 0; i < argc; ++i) {
    args_.push_back(std::string(argv[i]));
  }
}

// Check and parse user input
bool Checker::Check(HyperParam& hyper_param) {
  // Do not have any args
  if (args_.size() == 1) {
    printf("%s\n", option_help().c_str());
    exit(0);
  }
  // Parse and check argument
  if (is_train_) {
    return check_train_options(hyper_param);
  } else {
    return check_inference_options(hyper_param);
  }
}

// Check options for training tasks
bool Checker::check_train_options(HyperParam& hyper_param) {
  bool bo = true;
  /*********************************************************
   *  Check the file path of training data                 *
   *********************************************************/
  if (FileExist(args_[1].c_str())) {
    hyper_param.train_set_file = std::string(args_[1]);
  } else {
    printf("[Error] Training data file: %s does not exist \n",
           args_[1].c_str());
    return false;
  }
  /*********************************************************
   *  Check each input argument                            *
   *********************************************************/
  StringList list(args_.begin()+2, args_.end());
  StrSimilar ss;
  for (int i = 0; i < list.size(); ) {
    if (list[i].compare("-s") == 0) {  // task type
      int value = atoi(list[i+1].c_str());
      if (value < 0 || value > 6) {
        printf("[Error] -s can only be [0 - 6] : \n"
               "  for classification task \n"
               "    0 -- logistic regression (LR) \n"
               "    1 -- linear support vectors machine (SVM) \n"
               "    2 -- factorization machines (FM) \n"
               "    3 -- field-aware factorization machines (FFM) \n"
               "  for regression task \n"
               "    4 -- linear regression (LR) \n"
               "    5 -- factorization machines (FM) \n"
               "    6 -- field-aware factorization machines (FFM) \n");
        bo = false;
      } else {
        if (value == 0) {
          hyper_param.loss_func = "cross-entropy";
          hyper_param.score_func = "linear";
        } else if (value == 1) {
          hyper_param.loss_func = "hinge";
          hyper_param.score_func = "linear";
        } else if (value == 2) {
          hyper_param.loss_func = "cross-entropy";
          hyper_param.score_func = "fm";
        } else if (value == 3) {
          hyper_param.loss_func = "cross-entropy";
          hyper_param.score_func = "ffm";
        } else if (value == 4) {
          hyper_param.loss_func = "sqaured";
          hyper_param.score_func = "linear";
        } else if (value == 5) {
          hyper_param.loss_func = "squared";
          hyper_param.score_func = "fm";
        } else if (value == 6) {
          hyper_param.loss_func = "squared";
          hyper_param.score_func = "ffm";
        }
      }
      i += 2;
    } else if (list[i].compare("-x") == 0) {  // metrics
      if (list[i+1].compare("acc") != 0 &&
          list[i+1].compare("prec") != 0 &&
          list[i+1].compare("recall") != 0 &&
          list[i+1].compare("f1") != 0 &&
          list[i+1].compare("auc") != 0 &&
          list[i+1].compare("mae") != 0 &&
          list[i+1].compare("mape") != 0 &&
          list[i+1].compare("none") != 0) {
        printf("[Error] Unknow metric : %s \n"
               " -x can only be 'acc', 'prec', 'recall', "
               "'f1', 'auc', 'mae', 'mape', and 'none' \n",
               list[i+1].c_str());
        bo = false;
      } else {
        hyper_param.metric = list[i+1];
      }
      i += 2;
    } else if (list[i].compare("-t") == 0) {  // test file
      if (FileExist(list[i+1].c_str())) {
        hyper_param.test_set_file = list[i+1];
      } else {
        printf("[Error] Test set file: %s dose not exists \n",
               list[i+1].c_str());
        bo = false;
      }
      i += 2;
    } else if (list[i].compare("-m") == 0) {  // model file path
      hyper_param.model_file = list[i+1];
      i += 2;
    } else if (list[i].compare("-l") == 0) {  // log file path
      hyper_param.log_file = list[i+1];
      i += 2;
    } else if (list[i].compare("-k") == 0) {  // latent factor
      int value = atoi(list[i+1].c_str());
      if (value <= 0) {
        printf("[Error] Illegal -k '%i' \n"
               " -k must be geater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.num_K = value;
      }
      i += 2;
    } else if (list[i].compare("-r") == 0) {  // learning rate
      real_t value = atof(list[i+1].c_str());
      if (value <= 0) {
        printf("[Error] Illegal -r : '%f' \n"
               " -r must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.learning_rate = value;
      }
      i += 2;
    } else if (list[i].compare("-b") == 0) {  // regu lambda
      real_t value = atof(list[i+1].c_str());
      if (value < 0) {
        printf("[Error] Illegal -b : '%f' \n"
               " -b must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.regu_lambda = value;
      }
      i += 2;
    } else if (list[i].compare("-u") == 0) {  // model scale fatcor
      real_t value = atof(list[i+1].c_str());
      if (value <= 0) {
        printf("[Error] Illegal -u : '%f' \n"
               " -u must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.model_scale = value;
      }
      i += 2;
    } else if (list[i].compare("-e") == 0) {  // number of epoch
      int value = atoi(list[i+1].c_str());
      if (value < 0) {
        printf("[Error] Illegal -e : '%i' \n"
               " -e must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.num_epoch = value;
      }
      i += 2;
    } else if (list[i].compare("-f") == 0) {  // number of folds
      int value = atoi(list[i+1].c_str());
      if (value < 0) {
        printf("[Error] Illegal -f : '%i' \n"
               " -f must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.num_folds = value;
      }
      i += 2;
    } else if (list[i].compare("--disk") == 0) {  // on-disk training
      hyper_param.early_stop = true;
      i += 1;
    } else if (list[i].compare("--cv") == 0) {  // cross-validation
      hyper_param.cross_validation = true;
      i += 1;
    } else if (list[i].compare("--es") == 0) {  // early-stop
      hyper_param.early_stop = true;
      i += 1;
    } else if (list[i].compare("--no-norm") == 0) {  // normalization
      hyper_param.norm = false;
      i += 1;
    } else if (list[i].compare("--quiet") == 0) {  // quiet
      hyper_param.quiet = true;
      i += 1;
    } else {  // no match
      std::string similar_str;
      ss.FindSimilar(list[i], menu_, similar_str);
      printf("[Error] Unknow argument '%s'\n"
             "  Do you mean '%s' ?\n",
             list[i].c_str(),
             similar_str.c_str());
      bo = false;
      if (list[i][1] == '-') {  // "--" options
        i += 1;
      } else {  // "-" options
        i += 2;
      }
    }
  }
  if (!bo) { return false; }
  /*********************************************************
   *  Check some warnings and conflict                     *
   *********************************************************/
  if (hyper_param.cross_validation &&
     !hyper_param.test_set_file.empty()) {
    printf("[Warning] -c (cross-validation) has be set, and "
           "xLearn will ignore the test file: %s \n",
           hyper_param.test_set_file.c_str());
    hyper_param.test_set_file.clear();
  }
  if (hyper_param.early_stop &&
      hyper_param.test_set_file.empty() &&
     !hyper_param.cross_validation) {
    printf("[Error] To use early-stop, you need to "
           "assign a test set via '-t' option, or using "
           "cross-validation. \n");
    exit(0);
  }
  if (hyper_param.cross_validation &&
      !hyper_param.model_file.empty()) {
    printf("[Warning] Training in cross-validation and will "
           "not dump the final model checkpoint. \n");
    hyper_param.model_file.clear();
  }
  if (hyper_param.cross_validation &&
      hyper_param.quiet) {
    printf("[Warning] Cannot use -quiet option in "
           "cross-validation. \n");
    hyper_param.quiet = false;
  }
  if (hyper_param.loss_func.compare("cross-entropy") == 0 ||
      hyper_param.loss_func.compare("hinge") == 0) {
    // for classification
    if (hyper_param.metric.compare("mae") == 0 ||
        hyper_param.metric.compare("mape") == 0) {
      printf("[Error] The -x: %s metric can only be used "
             "in regression tasks. \n",
             hyper_param.metric.c_str());
      exit(0);
    }
  } else if (hyper_param.loss_func.compare("squared") == 0) {
    // for regression
    if (hyper_param.metric.compare("acc") == 0 ||
        hyper_param.metric.compare("prec") == 0 ||
        hyper_param.metric.compare("recall") == 0 ||
        hyper_param.metric.compare("f1") == 0) {
      printf("[Error] The -x: %s metric can only be used "
             "in classification tasks. \n",
              hyper_param.metric.c_str());
      exit(0);
    }
  }

  return true;
}

// Check options for inference tasks
bool Checker::check_inference_options(HyperParam& hyper_param) {
  bool bo = true;
  /*********************************************************
   *  Check the path of predict file                       *
   *********************************************************/
  if (FileExist(args_[1].c_str())) {
    hyper_param.predict_file = std::string(args_[1]);
  } else {
    printf("[Error] Predict data file: %s does not exist \n",
           args_[1].c_str());
    return false;
  }
  /*********************************************************
   *  Check the number of args                             *
   *********************************************************/
  StringList list(args_.begin()+2, args_.end());
  if (list.size() % 2 != 0) {
    printf("[Error] Every options should have a value \n");
    for (int i = 0; i < list.size(); i+=2) {
      printf("  %s : %s\n", list[i].c_str(), list[i+1].c_str());
    }
    return false;
  }
  /*********************************************************
   *  Check each input argument                            *
   *********************************************************/
  StrSimilar ss;
  for (int i = 0; i < list.size(); i+=2) {
    if (list[i].compare("-m") == 0) {  // path of the model file
      if (FileExist(list[i+1].c_str())) {
        hyper_param.model_file = list[i+1];
      } else {
        printf("[Error] Model file: %s does not exist \n",
               list[i+1].c_str());
        bo = false;
      }
    } else if (list[i].compare("-o") == 0) {  // path of the output
      hyper_param.output_file = list[i+1];
    } else if (list[i].compare("-l") == 0) {  // path of the log file
      hyper_param.log_file = list[i+1];
    } else {  // no match
      std::string similar_str;
      ss.FindSimilar(list[i], menu_, similar_str);
      printf("[Error] Unknow argument '%s'\n"
             " Do you mean '%s' ?\n",
             list[i].c_str(),
             similar_str.c_str());
      bo = false;
    }
  }
  if (!bo) { return false; }

  return true;
}

} // namespace xLearn
