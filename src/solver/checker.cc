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
#include "src/base/format_print.h"
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
"     xlearn_train <train_file_path> [OPTIONS] \n"
"                                                    \n"
" e.g.,  xlearn_train train_data.txt -s 0 -v validate_data.txt -r 0.1 \n"
"                                                                      \n"
"OPTIONS: \n"
"  -s <type> : Type of machine learning model (default 0) \n"
"     for classification task: \n"
"         0 -- linear model (GLM) \n"
"         1 -- factorization machines (FM) \n"
"         2 -- field-aware factorization machines (FFM) \n"
"     for regression task: \n"
"         3 -- linear model (GLM) \n"
"         4 -- factorization machines (FM) \n"
"         5 -- field-aware factorization machines (FFM) \n"
"                                                                            \n"
"  -x <metric>          :  The metric can be 'acc', 'prec', 'recall', 'f1' (classification), and 'mae',\n"
"                          'mape', 'rmsd (rmse)' (regression). xLearn uses the Accuracy (acc) by default.\n"
"                          If we set this option to 'none', xLearn will not print any metric information.\n"
"                                                                                                 \n"
"  -v <validate_file>   :  Path of the validation data file. This option will be empty by default, \n"
"                          and in this way, the xLearn will not perform validation. \n"
"                                                                                              \n"
"  -m <model_file>      :  Path of the model checkpoint file. On default, the model file name will be. \n"
"                          set to 'train_file' + '.model'. If we set this value to 'none', the xLearn will \n"
"                          not dump the model checkpoint after training. \n"
"                                                                             \n"
"  -l <log_file>        :  Path of the log file. Using '/tmp/xlearn_log/' by default. \n"
"                                                                                       \n"
"  -k <number_of_K>     :  Number of the latent factor used by fm and ffm tasks. Using 4 by default. \n"
"                          Note that, we will get the same model size when setting k to 1 and 4. \n"
"                          This is because we use SSE instruction and the memory need to be aligned. \n"
"                          So even you assign k = 1, we still fill some dummy zeros from k = 2 to 4. \n"
"                                                                                         \n"
"  -r <learning_rate>   :  Learning rate for stochastic gradient descent. Using 0.2 by default. \n"
"                          xLearn uses adaptive gradient descent (AdaGrad) for optimization problem, \n"
"                          and the learning rate will be changed adaptively. \n"
"                                                                                     \n"
"  -b <lambda_for_regu> :  Lambda for L2 regular. Using 0.00002 by default. We can disable the \n"
"                          regular term by setting this value to 0.0 \n"
"                                                                      \n"
"  -u <model_scale>     :  Hyper parameter used for initialize model parameters. \n"
"                          Using 0.66 by default. \n"
"                                                                                  \n"
"  -e <epoch_number>    :  Number of epoch for training. Using 10 by default. Note that, xLearn will \n"
"                          perform early-stopping by default, so this value is just a upper bound. \n"
"                                                                                       \n"
"  -f <fold_number>     :  Number of folds for cross-validation. Using 5 by default.      \n"
"                                                                                      \n"
"  --disk               :  Open on-disk training for large-scale machine learning problems. \n"
"                                                                    \n"
"  --cv                 :  Open cross-validation in training tasks. If we use this option, xLearn \n"
"                          will ignore the validation file (-t).  \n"
"                                                                   \n"
"  --lock-free          :  Open lock-free training. Lock-free training can accelerate training but \n"
"                          the result is non-deterministic. Our suggestion is that you can open this flag \n"
"                          if the training data is bigger than 200MB. \n"
"                                                                        \n"
"  --dis-es             :  Disable early-stopping in training. By default, xLearn will use early-stopping \n"
"                          in training tasks, except for training in cross-validation. \n"
"                                                                                          \n"
"  --no-norm            :  Disable instance-wise normalization. By default, xLearn will use \n"
"                          instance-wise normalization for both training and prediction. \n"
"                                                                  \n"
"  --quiet              :  Don't print any evaluation information during the training and \n"
"                          just train the model quietly. \n"
"----------------------------------------------------------------------------------------------\n"
    );
  } else {
    return std::string(
"-------------------------------------- Prediction task ---------------------------------------\n"
"USAGE: \n"
"     xlearn_predict <test_file> <model_file> [OPTIONS] \n"
"                                                         \n"
" e.g.,  xlearn_predict ./test_data.txt ./model_file -o ./out.txt  \n"
"                                                                           \n"
"OPTIONS: \n"
"  -o <output_file>     :  Path of the output file. On default, this value will be set \n"
"                          to 'test_file' + '.out'     \n"
"                                                             \n"
"  -l <log_file_path>   :  Path of the log file. Using '/tmp/xlearn_log' by default. \n"
"----------------------------------------------------------------------------------------------\n"
    );
  }
}

// Initialize Checker
void Checker::Initialize(bool is_train, int argc, char* argv[]) {
  is_train_ = is_train;
  if (is_train_) {  // for training
    menu_.push_back(std::string("-s"));
    menu_.push_back(std::string("-x"));
    menu_.push_back(std::string("-v"));
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
    menu_.push_back(std::string("--dis-es"));
    menu_.push_back(std::string("--no-norm"));
    menu_.push_back(std::string("--quiet"));
  } else {  // for Prediction
    menu_.push_back(std::string("-o"));
    menu_.push_back(std::string("-l"));
    menu_.push_back(std::string("--sign"));
    menu_.push_back(std::string("--sigmoid"));
  }
  // Get the user's input
  for (int i = 0; i < argc; ++i) {
    args_.push_back(std::string(argv[i]));
  }
}

// Check and parse user's input
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
    return check_prediction_options(hyper_param);
  }
}

// Check options for training tasks
bool Checker::check_train_options(HyperParam& hyper_param) {
  bool bo = true;
  /*********************************************************
   *  Check the file path of the training data             *
   *********************************************************/
  if (FileExist(args_[1].c_str())) {
    hyper_param.train_set_file = std::string(args_[1]);
  } else {
    print_error(
      StringPrintf("Training data file: %s does not exist.", 
                    args_[1].c_str())
    );
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
      if (value < 0 || value > 5) {
        print_error(
            "-s can only be [0 - 5] : \n"
            "  for classification task: \n"
            "    0 -- linear model (GLM) \n"
            "    1 -- factorization machines (FM) \n"
            "    2 -- field-aware factorization machines (FFM) \n"
            "  for regression task: \n"
            "    3 -- linear model (GLM) \n"
            "    4 -- factorization machines (FM) \n"
            "    5 -- field-aware factorization machines (FFM)");
        bo = false;
      } else {
        switch (value) {
          case 0:
            hyper_param.loss_func = "cross-entropy";
            hyper_param.score_func = "linear";
            break;
          case 1:
            hyper_param.loss_func = "cross-entropy";
            hyper_param.score_func = "fm";
            break;
          case 2:
            hyper_param.loss_func = "cross-entropy";
            hyper_param.score_func = "ffm";
            break;
          case 3:
            hyper_param.loss_func = "squared";
            hyper_param.score_func = "linear";
            break;
          case 4:
            hyper_param.loss_func = "squared";
            hyper_param.score_func = "fm";
            break;
          case 5:
            hyper_param.loss_func = "squared";
            hyper_param.score_func = "ffm";
            break;
          default: break;
        }
      }
      i += 2;
    } else if (list[i].compare("-x") == 0) {  // metrics
      if (list[i+1].compare("acc") != 0 &&
          list[i+1].compare("prec") != 0 &&
          list[i+1].compare("recall") != 0 &&
          list[i+1].compare("f1") != 0 &&
          list[i+1].compare("mae") != 0 &&
          list[i+1].compare("mape") != 0 &&
          list[i+1].compare("rmsd") != 0 &&
          list[i+1].compare("rmse") != 0 &&
          list[i+1].compare("none") != 0) {
        print_error(
          StringPrintf("Unknow metric: %s \n"
               " -x can only be: \n"
               "   acc \n"
               "   prec \n" 
               "   recall \n"
               "   f1 \n"
               "   mae \n"
               "   mape \n"
               "   rmsd \n"
               "   rmse \n"
               "   none",
               list[i+1].c_str())
        );
        bo = false;
      } else {
        hyper_param.metric = list[i+1];
      }
      i += 2;
    } else if (list[i].compare("-v") == 0) {  // validation file
      if (FileExist(list[i+1].c_str())) {
        hyper_param.validate_set_file = list[i+1];
      } else {
        print_error(
          StringPrintf("Validation set file: %s dose not exists.",
                       list[i+1].c_str())
        );
        bo = false;
      }
      i += 2;
    } else if (list[i].compare("-m") == 0) {  // model file
      hyper_param.model_file = list[i+1];
      i += 2;
    } else if (list[i].compare("-l") == 0) {  // log file
      hyper_param.log_file = list[i+1];
      i += 2;
    } else if (list[i].compare("-k") == 0) {  // latent factor
      int value = atoi(list[i+1].c_str());
      if (value <= 0) {
        print_error(
          StringPrintf("Illegal -k '%i' \n"
               " -k must be geater than zero.",
               value)
        );
        bo = false;
      } else {
        hyper_param.num_K = value;
      }
      i += 2;
    } else if (list[i].compare("-r") == 0) {  // learning rate
      real_t value = atof(list[i+1].c_str());
      if (value <= 0) {
        print_error(
          StringPrintf("Illegal -r : '%f' \n"
               " -r must be greater than zero.",
               value)
        );
        bo = false;
      } else {
        hyper_param.learning_rate = value;
      }
      i += 2;
    } else if (list[i].compare("-b") == 0) {  // regular lambda
      real_t value = atof(list[i+1].c_str());
      if (value < 0) {
        print_error(
          StringPrintf("Illegal -b : '%f' \n"
               " -b must be greater than or equal to zero.",
               value)
        );
        bo = false;
      } else {
        hyper_param.regu_lambda = value;
      }
      i += 2;
    } else if (list[i].compare("-u") == 0) {  // model scale
      real_t value = atof(list[i+1].c_str());
      if (value <= 0) {
        print_error(
          StringPrintf("Illegal -u : '%f' \n"
               " -u must be greater than zero.",
               value)
        );
        bo = false;
      } else {
        hyper_param.model_scale = value;
      }
      i += 2;
    } else if (list[i].compare("-e") == 0) {  // number of epoch
      int value = atoi(list[i+1].c_str());
      if (value < 0) {
        print_error(
          StringPrintf("Illegal -e : '%i' \n"
               " -e must be greater than zero.",
               value)
        );
        bo = false;
      } else {
        hyper_param.num_epoch = value;
      }
      i += 2;
    } else if (list[i].compare("-f") == 0) {  // number of folds
      int value = atoi(list[i+1].c_str());
      if (value < 0) {
        print_error(
          StringPrintf("Illegal -f : '%i' \n"
               " -f must be greater than zero.",
               value)
        );
        bo = false;
      } else {
        hyper_param.num_folds = value;
      }
      i += 2;
    } else if (list[i].compare("--disk") == 0) {  // on-disk training
      hyper_param.on_disk = true;
      i += 1;
    } else if (list[i].compare("--cv") == 0) {  // cross-validation
      hyper_param.cross_validation = true;
      i += 1;
    } else if (list[i].compare("--lock-free") == 0) {  // lock-free training
      hyper_param.lock_free = true;
      i += 1;
    } else if (list[i].compare("--dis-es") == 0) {  // disable early-stop
      hyper_param.early_stop = false;
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
      print_error(
        StringPrintf("Unknow argument '%s'\n"
             "  Do you mean '%s' ?",
             list[i].c_str(),
             similar_str.c_str())
      );
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
   *  Check warning and fix conflict                       *
   *********************************************************/
  if (hyper_param.on_disk && hyper_param.cross_validation) {
    print_warning("On-disk training doesn't support cross-validation. "
                  "xLearn has already disable the -cv option.");
    hyper_param.cross_validation = false;
  }
  if (hyper_param.cross_validation && hyper_param.early_stop) {
    print_warning("Cross-validation doesn't support early-stopping. "
                  "xLearn has already close early-stopping.");
    hyper_param.early_stop = false;
  }
  if (hyper_param.cross_validation && !hyper_param.test_set_file.empty()) {
    print_warning(
      StringPrintf("The --cv (cross-validation) has been set, and "
                   "xLearn will ignore the validation file: %s",
                   hyper_param.test_set_file.c_str())
    );
    hyper_param.validate_set_file.clear();
  }
  if (hyper_param.cross_validation && hyper_param.quiet) {
    print_warning("The --cv (cross-validation) has been set, and "
                  "xLearn will ignore the --quiet option.");
    hyper_param.quiet = false;
  }
  if (hyper_param.cross_validation && !hyper_param.model_file.empty()) {
    print_warning("The --cv (cross-validation) has been set, and "
                  "xLearn will not dump model checkpoint to disk.");
    hyper_param.model_file.clear();
  }
  if (hyper_param.validate_set_file.empty() && hyper_param.early_stop) {
    print_warning("Validation file not found, xLearn has already "
                  "disable early-stopping.");
    hyper_param.early_stop = false;
  }
  if (hyper_param.metric.compare("none") != 0 &&
      hyper_param.validate_set_file.empty() &&
      !hyper_param.cross_validation) {
    print_warning(
      StringPrintf("Validation file not found, xLearn has already "
                   "disable (-x %s) option.", 
                   hyper_param.metric.c_str())
    );
    hyper_param.metric = "none";
  }
  if (hyper_param.loss_func.compare("squared") == 0) {
    if (hyper_param.metric.compare("acc") == 0 ||
        hyper_param.metric.compare("prec") == 0 ||
        hyper_param.metric.compare("recall") == 0 ||
        hyper_param.metric.compare("f1") == 0) {
      print_warning(
        StringPrintf("The -x: %s metric can only be used "
                     "in classification tasks. xLearn will "
                     "ignore this option.",
                     hyper_param.metric.c_str())
      );
      hyper_param.metric = "none";
    }
  } else if (hyper_param.loss_func.compare("cross-entropy") == 0) {
    if (hyper_param.metric.compare("mae") == 0 ||
        hyper_param.metric.compare("mape") == 0 ||
        hyper_param.metric.compare("rmsd") == 0 ||
        hyper_param.metric.compare("rmse") == 0) {
      print_warning(
        StringPrintf("The -x: %s metric can only be used "
                     "in regression tasks. xLearn will ignore "
                     "this option.",
                     hyper_param.metric.c_str())
        );
      hyper_param.metric = "none";
    }
  }
  /*********************************************************
   *  Set default value                                    *
   *********************************************************/
  if (hyper_param.model_file.empty() && !hyper_param.cross_validation) {
    hyper_param.model_file = hyper_param.train_set_file + ".model";
  }
  if (hyper_param.metric.compare("rmse") == 0) {
    hyper_param.metric = "rmsd";
  }

  return true;
}

// Check options for prediction tasks
bool Checker::check_prediction_options(HyperParam& hyper_param) {
  bool bo = true;
  /*********************************************************
   *  Check size                                           *
   *********************************************************/
  if (args_.size() < 3) {
    print_error("The test file and model file must be set.");
    return false;
  }
  /*********************************************************
   *  Check the path of test set file                      *
   *********************************************************/
  if (FileExist(args_[1].c_str())) {
    hyper_param.test_set_file = std::string(args_[1]);
  } else {
    print_error(
      StringPrintf("Test set file: %s does not exist.",
           args_[1].c_str())
    );
    return false;
  }
  /*********************************************************
   *  Check the path of model file                         *
   *********************************************************/
  if (FileExist(args_[2].c_str())) {
    hyper_param.model_file = std::string(args_[2]);
  } else {
    print_error(
      StringPrintf("Model file: %s does not exist.",
           args_[2].c_str())
    );
    return false;
  }
  /*********************************************************
   *  Check each input argument                            *
   *********************************************************/
  StringList list(args_.begin()+3, args_.end());
  StrSimilar ss;
  for (int i = 0; i < list.size(); ) {
    if (list[i].compare("-o") == 0) {  // path of the output
      hyper_param.output_file = list[i+1];
      i += 2;
    } else if (list[i].compare("-l") == 0) {  // path of the log file
      hyper_param.log_file = list[i+1];
      i += 2;
    } else if (list[i].compare("--sign") == 0) {  // convert output to 0 and 1
      hyper_param.sign = true;
      i += 1;
    } else if (list[i].compare("--sigmoid") == 0) {  // using sigmoid
      hyper_param.sigmoid = true;
      i += 1;
    } else {  // no match
      std::string similar_str;
      ss.FindSimilar(list[i], menu_, similar_str);
      print_error(
        StringPrintf("Unknow argument '%s'\n"
             " Do you mean '%s' ?",
             list[i].c_str(),
             similar_str.c_str())
      );
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
   *  Check warning and fix conflict                       *
   *********************************************************/
  if (hyper_param.sign && hyper_param.sigmoid) {
    print_warning("Both of --sign and --sigmoid have been set. "
                  "xLearn has already disable --sign and --sigmoid.");
    hyper_param.sign = false;
    hyper_param.sigmoid = false;
  }
  /*********************************************************
   *  Set default value                                    *
   *********************************************************/
  if (hyper_param.output_file.empty()) {
    hyper_param.output_file = hyper_param.test_set_file + ".out";
  }

  return true;
}

} // namespace xLearn
