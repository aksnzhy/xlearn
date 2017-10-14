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
"--------------------------------  Train task  ------------------------------\n"
"USAGE: \n"
"     xlearn_train [ train_file_path ] [ OPTIONS ] \n"
"                                           \n"
"OPTIONS: \n"
"  -s <type> : set type of machine learning model (default 0) \n"
"     for classification task \n"
"         0 -- logistic regression (LR) \n"
"         1 -- linear support vectors machine (SVM) \n"
"         2 -- factorization machines (FM) \n"
"         3 -- field-aware factorization machines (FFM) \n"
"     for regression task \n"
"         4 -- linear regression (LR) \n"
"         5 -- factorization machines (FM) \n"
"         6 -- field-aware factorization machines (FFM) \n"
"                                                                 \n"
"  -d <true_or_false>   :  Training on disk for limited memory \n"
"                          (use 'false' by default to train in memory) \n"
"  -x <metric>          :  Evaluation metric can be 'acc', 'prec', \n"
"                          'recall', 'roc', 'auc', 'mae', 'mse' \n"
"                          (use 'acc' - Accuracy by default) \n"
"  -t <test_file_path>  :  Path of the test data file \n"
"                          (this option can be empty by default) \n"
"  -m <model_file_path> :  Path of the model checkpoint file \n"
"                          (use './xlearn_model' by default) \n"
"  -l <log_file_path>   :  Path of the log file \n"
"                          (use './xlearn_log' by default) \n"
"  -k <number_of_K>     :  Number of the latent factor for fm and ffm. \n"
"                          Note that -k must be a multiple of 8 like 8, 16 .. \n"
"                          (use 8 by default) \n"
"  -r <learning_rate>   :  Learning rate for gradient descent \n"
"                          (use 0.0005 by default) \n"
"  -b <lambda_for_regu> :  Lambda for regular and use 0 to close the regular \n"
"                          (use 0.00002 by default) \n"
"  -e <epoch_number>    :  Number of epoch for training \n"
"                          (use 10 by default) \n"
"  -c <true_or_false>   :  Using cross-validation \n"
"                          (use 'false' by default) \n"
"  -f <fold_number>     :  Fold number for cross-validation \n"
"                          (use 5 by default) \n"
"  -p <true_or_false>   :  Using early-stop in training \n"
"                          (use 'false' by default) \n"
"----------------------------------------------------------------------------\n"
    );
  } else {
    return std::string(
"------------------------------ Predict task --------------------------------\n"
"USAGE: \n"
"     xlearn_predict [ predict_data_path ] [ options ] \n"
"                                                     \n"
"OPTIONS: \n"
"  -m <model_file_path>  :  Path of the trained model file \n"
"                           (use './xlearn_model' by default) \n"
"  -o <output_file_path> :  Path of the output file \n"
"                           (use './xlearn_out' by default) \n"
"  -l <log_file_path>    :  Path of the log file \n"
"                           (use './xlearn_log' by default) \n"
"----------------------------------------------------------------------------\n"
    );
  }
}

// Convert upercase to lowcase
char easy_to_lower(char in) {
  if(in <= 'Z' && in >= 'A') {
    return in - ('Z'-'z');
  }
  return in;
}

// Initialize Checker
void Checker::Initialize(bool is_train, int argc, char* argv[]) {
  is_train_ = is_train;
  if (is_train_) {
    menu_.push_back(std::string("-s"));
    menu_.push_back(std::string("-d"));
    menu_.push_back(std::string("-x"));
    menu_.push_back(std::string("-t"));
    menu_.push_back(std::string("-m"));
    menu_.push_back(std::string("-l"));
    menu_.push_back(std::string("-k"));
    menu_.push_back(std::string("-r"));
    menu_.push_back(std::string("-b"));
    menu_.push_back(std::string("-e"));
    menu_.push_back(std::string("-c"));
    menu_.push_back(std::string("-f"));
    menu_.push_back(std::string("-p"));
  } else {  // for Predict
    menu_.push_back(std::string("-m"));
    menu_.push_back(std::string("-o"));
    menu_.push_back(std::string("-l"));
  }
  // Get the user input
  for (int i = 0; i < argc; ++i) {
    args_.push_back(std::string(argv[i]));
  }
  // Convert args to lower case
  // skip the program name and file path
  for (int i = 2; i < args_.size(); ++i) {
    std::transform(args_[i].begin(),
                   args_[i].end(),
                   args_[i].begin(),
                   easy_to_lower);
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
   *  Step 1: Check the path of train file                 *
   *********************************************************/
  if (FileExist(args_[1].c_str())) {
    hyper_param.train_set_file = std::string(args_[1]);
  } else {
    printf("[Error] Training data file: %s does not exist \n",
           args_[1].c_str());
    return false;
  }
  /*********************************************************
   *  Step 2: Check the number of args                     *
   *********************************************************/
  StringList list(args_.begin()+2, args_.end());
  if (list.size() % 2 != 0) {
    printf("[Error] Every options should have its value \n");
    for (int i = 0; i < list.size(); i+=2) {
      printf("  %s : %s\n", list[i].c_str(), list[i+1].c_str());
    }
    return false;
  }
  /*********************************************************
   *  Step 3: Check each argument                          *
   *********************************************************/
  StrSimilar ss;
  for (int i = 0; i < list.size(); i += 2) {
    if (list[i].compare("-s") == 0) {
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
    } else if (list[i].compare("-d") == 0) {
      if (list[i+1].compare("true") != 0 &&
          list[i+1].compare("false") != 0) {
        printf("[Error] -d can only be 'true' or 'false' : %s\n",
               list[i+1].c_str());
        bo = false;
      } else {
        hyper_param.on_disk = list[i+1] == "true" ? true : false;
      }
    } else if (list[i].compare("-x") == 0) {
      if (list[i+1].compare("acc") != 0 &&
          list[i+1].compare("prec") != 0 &&
          list[i+1].compare("recall") != 0 &&
          list[i+1].compare("roc") != 0 &&
          list[i+1].compare("auc") != 0 &&
          list[i+1].compare("mae") != 0 &&
          list[i+1].compare("mse") != 0) {
        printf("[Error] Unknow metric : %s \n"
               " -x can only be 'acc', 'prec', 'recall', "
               "'roc', 'auc', 'mae', 'mse' \n", list[i+1].c_str());
        bo = false;
      } else {
        hyper_param.metric = list[i+1];
      }
    } else if (list[i].compare("-t") == 0) {
      if (FileExist(list[i+1].c_str())) {
        hyper_param.test_set_file = list[i+1];
      } else {
        printf("[Error] Test set file: %s dose not exists \n",
               list[i+1].c_str());
        bo = false;
      }
    } else if (list[i].compare("-m") == 0) {
      hyper_param.model_file = list[i+1];
    } else if (list[i].compare("-l") == 0) {
      hyper_param.log_file = list[i+1];
    } else if (list[i].compare("-k") == 0) {
      int value = atoi(list[i+1].c_str());
      if (value % 8 != 0) {
        printf("[Error] Illegal -k '%i' \n"
               " -k must be a multiple of 8, like 8, 16 32 ... \n",
               value);
        bo = false;
      } else {
        hyper_param.num_K = value;
      }
    } else if (list[i].compare("-r") == 0) {
      real_t value = atof(list[i+1].c_str());
      if (value <= 0) {
        printf("[Error] Illegal -r : '%f' \n"
               " -r must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.learning_rate = value;
      }
    } else if (list[i].compare("-b") == 0) {
      real_t value = atof(list[i+1].c_str());
      if (value < 0) {
        printf("[Error] Illegal -b : '%f' \n"
               " -b must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.regu_lambda = value;
      }
    } else if (list[i].compare("-e") == 0) {
      int value = atoi(list[i+1].c_str());
      if (value < 0) {
        printf("[Error] Illegal -e : '%i' \n"
               " -e must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.num_epoch = value;
      }
    } else if (list[i].compare("-c") == 0) {
      std::string value = list[i+1];
      if (value.compare("true") != 0 &&
          value.compare("false") != 0) {
        printf("[Error] Illegal -c : '%s' \n"
               " -c can only be 'true' or 'false' \n",
               value.c_str());
        bo = false;
      } else {
        hyper_param.cross_validation =
          value.compare("true") == 0 ? true : false;
      }
    } else if (list[i].compare("-f") == 0) {
      int value = atoi(list[i+1].c_str());
      if (value < 0) {
        printf("[Error] Illegal -f : '%i' \n"
               " -f must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.num_folds = value;
      }
    } else if (list[i].compare("-p") == 0) {
      std::string value = list[i+1];
      if (value.compare("true") != 0 &&
          value.compare("false") != 0) {
        printf("[Error] Illegal -p : '%s' \n"
               " -p can only be 'true' or 'false' \n",
               value.c_str());
        bo = false;
      } else {
        hyper_param.early_stop = value == "true" ? true : false;
      }
    } else { // no option match
      std::string similar_str;
      // not very similar
      if (ss.FindSimilar(list[i], menu_, similar_str) > 3) {
        printf("[Error] Unknow argument '%s'\n", list[i].c_str());
      } else {
        printf("[Error] Unknow argument '%s'\n"
               "  Do you mean '%s' ?\n",
               list[i].c_str(),
               similar_str.c_str());
      }
      bo = false;
    }
  }
  if (!bo) { return false; }
  /*********************************************************
   *  Step 4: Check some warnings and conflict             *
   *********************************************************/
  if (hyper_param.cross_validation &&
     !hyper_param.test_set_file.empty()) {
    printf("[Warning] -c has be set, and xLearn will "
           "ignore the test file: %s \n",
           hyper_param.test_set_file.c_str());
  }
  if (hyper_param.num_K > 80) {
    printf("[Warning] -k is too large: %d \n",
           hyper_param.num_K);
  }
  if (hyper_param.learning_rate > 2.0) {
    printf("[Warning] -r is too large: %f \n",
           hyper_param.learning_rate);
  }
  if (hyper_param.regu_lambda > 0.5) {
    printf("[Warning] -b is too large: %f \n",
           hyper_param.regu_lambda);
  }
  if (hyper_param.num_folds > 10) {
    printf("[Warning] -f is too large: %d \n",
           hyper_param.num_folds);
  }
  if (hyper_param.num_epoch > 1000) {
    printf("[Warning] -e is too large: %d \n",
           hyper_param.num_epoch);
  }
  if (hyper_param.early_stop &&
      hyper_param.test_set_file.empty() &&
     !hyper_param.cross_validation) {
    printf("[Error] To use early-stop, you need to "
           "assign a test set via '-t' option \n");
    exit(0);
  }

  return true;
}

// Check options for inference tasks
bool Checker::check_inference_options(HyperParam& hyper_param) {
  bool bo = true;
  /*********************************************************
   *  Step 1: Check the path of predict file               *
   *********************************************************/
  if (FileExist(args_[1].c_str())) {
    hyper_param.predict_file = std::string(args_[1]);
  } else {
    printf("[Error] Predict data file: %s does not exist \n",
           args_[1].c_str());
    return false;
  }
  /*********************************************************
   *  Step 2: Check the number of args                     *
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
   *  Step 3: Check each argument                          *
   *********************************************************/
  StrSimilar ss;
  for (int i = 0; i < list.size(); i+=2) {
    if (list[i].compare("-m") == 0) {
      if (FileExist(list[i+1].c_str())) {
        hyper_param.model_file = list[i+1];
      } else {
        printf("[Error] Model file: %s does not exist \n",
               list[i+1].c_str());
        bo = false;
      }
    } else if (list[i].compare("-o") == 0) {
      hyper_param.output_file = list[i+1];
    } else if (list[i].compare("-l") == 0) {
      hyper_param.log_file = list[i+1];
    } else {  // no option match
      std::string similar_str;
      // not very similar
      if (ss.FindSimilar(list[i], menu_, similar_str) > 3) {
        printf("[Error] Unknow argument '%s'\n", list[i].c_str());
      } else {
        printf("[Error] Unknow argument '%s'\n"
               " Do you mean '%s' ?\n",
               list[i].c_str(),
               similar_str.c_str());
      }
      bo = false;
    }
  }
  if (!bo) { return false; }

  return true;
}

} // namespace xLearn
