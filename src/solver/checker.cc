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

#include "src/solver/checker.h"
#include "src/base/levenshtein_distance.h"

namespace xLearn {

// Option help menu
std::string Checker::option_help() const {
  return std::string(
"--------------------------------  Train task  ------------------------------\n"
"Usage: xlearn --is_train [options] \n"
"\n"
"options: \n\n"
/* must have value */
"    -train_data <path>: path of your training data file \n"
"    -test_data <path> (optional): path of your validation data file \n"
"    -model_file <path> (optional, use './xlearn-model' by default): \n"
"                       path of your model checkpoint file \n"
/* must have value */
"    -score <score_func>: score_func can be 'linear', 'fm', and 'ffm' \n"
/* must have value */
"    -loss <loss_func>: loss_function can be 'squared', 'absolute', \n"
"                       'cross_entropy', and 'hinge' \n"
"    -regular <regular> (optional, use 'none' by default): \n"
"                       regular can be 'l1', 'l2', 'l1_l2', and 'none' \n"
"    -updater <updater_method> (optional, use 'sgd' by default): \n"
"                              updater can be 'sgd', 'adagrad', 'adam' \n"
"                              'adadelta', 'rmsprop', and 'momentum' \n"
"    -k <value of factor> (optional, use 8 by default): \n"
"                         number of latent factor for fm and ffm. \n"
"                         Note that is value must be a multiple of 8 \n"
"    -lr <learning_rate> (optional, use 0.03 by default): \n"
"                        learning rate for optimization method \n"
"    -decay_rate <decay_rate> (optional, use 0.1 by default): \n"
"                              decay rate for optimization method \n"
"    -second_decay_rate <decay_rate> (optional, use 0.1 by default): \n"
"                                     second_decay_rate for optimization \n"
"    -regu_lambda_1 <lambda> (optional, use 0.03 by default): \n"
"                            lambda_1 for regular term \n"
"    -regu_lambda_2 <lambda> (optional, use 0.03 by default): \n"
"                            lambda_2 for regular term \n"
"    -epoch <number> (optional, use 10 by default): \n"
"                    number of epoch for training \n"
"    -batch_size <size> (optional, use 200 by default): \n"
"                       batch size for each training \n"
"    -file_format <format> (optional, use 'libsvm' or 'libffm' by default): \n"
"                          format can be 'libsvm', 'libffm', and 'csv' \n"
"    -cv <true or false> (optional, use false by default): \n"
"                         using cross-validation in training \n"
"    -fold <number> (optional, use 5 by default): \n"
"                   fold number for cross-validation \n"
"    -early_stop <true or false> (optional, use false by default): \n"
"                                using early-stop in training \n"
"    -on_disk <true or false> (optional, use false by default): \n"
"                             using on-disk training for limited memory \n"
"    -log_file <path> (optional, use './xlearn_log' by default): \n"
"                     path of your log file. \n"
"----------------------------------------------------------------------------\n"
"\n"
"----------------------------- Inference task -------------------------------\n"
"Usage: xlearn --is_inference [options] \n"
"\n"
"options: \n\n"
"   -model_file <path> (optional, use './xlearn_model' by default): \n"
"                      path of your pre-trained model file \n"
/* must have value */
"   -infer_data <path>: path of your inference data file \n"
"   -out_data <path> (optional, use './xlearn_out' by default): \n"
"                    path of your output file \n"
"----------------------------------------------------------------------------\n"
  );
}

// Convert uper case to lower case
char easy_to_lower(char in) {
  if(in <= 'Z' && in >= 'A')
    return in - ('Z'-'z');
  return in;
}

// Initialize Checker
void Checker::Initialize(int argc, char* argv[]) {
  // All the possible options
  menu_.push_back(std::string("-train_data"));
  menu_.push_back(std::string("-test_data"));
  menu_.push_back(std::string("-model_file"));
  menu_.push_back(std::string("-score"));
  menu_.push_back(std::string("-loss"));
  menu_.push_back(std::string("-regular"));
  menu_.push_back(std::string("-updater"));
  menu_.push_back(std::string("-k"));
  menu_.push_back(std::string("-lr"));
  menu_.push_back(std::string("-decay_rate"));
  menu_.push_back(std::string("-second_decay_rate"));
  menu_.push_back(std::string("-regu_lambda_1"));
  menu_.push_back(std::string("-regu_lambda_2"));
  menu_.push_back(std::string("-epoch"));
  menu_.push_back(std::string("-batch_size"));
  menu_.push_back(std::string("-file_format"));
  menu_.push_back(std::string("-cv"));
  menu_.push_back(std::string("-fold"));
  menu_.push_back(std::string("-early_stop"));
  menu_.push_back(std::string("-on_disk"));
  menu_.push_back(std::string("-infer_data"));
  menu_.push_back(std::string("-out_data"));
  // Get the user input
  for (int i = 0; i < argc; ++i) {
    args_.push_back(std::string(argv[i]));
  }
  // Convert args to lower case
  for (int i = 0; i < args_.size(); ++i) {
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
  if (args_[1].compare("--is_train") == 0) {
    hyper_param.is_train = true;
    return check_train_options(hyper_param);
  } else if (args_[1].compare("--is_inference") == 0) {
    hyper_param.is_train = false;
    return check_inference_options(hyper_param);
  } else {
    printf("[Error] Please use: \n"
           " 'xlearn --is_train [options]' for training task \n"
           "or \n 'xlearn --is_inference [options]' for inference task \n"
           "Type 'xlearn' for the details of the [options] \n");
    return false;
  }
}

// Check options for training tasks
bool Checker::check_train_options(HyperParam& hyper_param) {
  bool bo = true;
  // Remove the first two arguments
  StringList list(args_.begin()+2, args_.end());
  StrSimilar ss;
  // Every option should have a value
  if (list.size() % 2 != 0) {
    printf("[Error] Every options should have a value \n");
    return false;
  }
  // User must set the -train_data
  if (!ss.Find(std::string("-train_data"), list)) {
    printf("[Error] User need to set the option [-train_data] "
           "to specify the input training data file. \n");
    bo = false;
  }
  // User must set the -score
  if (!ss.Find(std::string("-score"), list)) {
    printf("[Error] User need to set the option [-score] "
           "to specify the score function, which can be 'linear', "
           "'fm', and 'ffm' \n");
    bo = false;
  }
  // User must set the -loss
  if (!ss.Find(std::string("-loss"), list)) {
    printf("[Error] User need to set the option [-loss] "
           "to specify the loss function, which can be 'squared', "
           "'absolute', 'cross_entropy', and 'hinge' \n");
    bo = false;
  }

  if (!bo) { return false; }

  // Check every single element
  for (int i = 0; i < list.size(); i+=2) {
    if (list[i].compare("-train_data") == 0) {
      hyper_param.train_set_file = list[i+1];
    } else if (list[i].compare("-test_data") == 0) {
      hyper_param.test_set_file = list[i+1];
    } else if (list[i].compare("-model_file") == 0) {
      hyper_param.model_file = list[i+1];
    } else if (list[i].compare("-score") == 0) {
      std::string value = list[i+1];
      if (value.compare("linear") != 0 &&
          value.compare("fm") != 0 &&
          value.compare("ffm") != 0) {
        printf("[Error] Unknow score function '%s' \n"
               " -score can only be 'linear', 'fm', or 'ffm' \n",
               value.c_str());
        bo = false;
      } else {
        hyper_param.score_func = value;
      }
    } else if (list[i].compare("-loss") == 0) {
      std::string value = list[i+1];
      if (value.compare("squared") != 0 &&
          value.compare("hinge") != 0 &&
          value.compare("cross-entropy") != 0 &&
          value.compare("absolute") != 0) {
        printf("[Error] Unknow loss function '%s' \n"
               " -loss can only be 'squared', "
               "'hinge', 'cross-entropy', or 'absolute' \n",
               value.c_str());
        bo = false;
      } else {
        hyper_param.loss_func = value;
      }
    } else if (list[i].compare("-regular") == 0) {
      std::string value = list[i+1];
      if (value.compare("l1") != 0 &&
          value.compare("l2") != 0 &&
          value.compare("l1_l2") != 0 &&
          value.compare("none") != 0) {
        printf("[Error] Unknow regular type: '%s' \n"
               " -regular can only be 'l1', 'l2', 'l1_l2', "
               "or 'none'\n",
               value.c_str());
        bo = false;
      } else {
        hyper_param.regu_type = value;
      }
    } else if (list[i].compare("-updater") == 0) {
      std::string value = list[i+1];
      if (value.compare("sgd") != 0 &&
          value.compare("adam") != 0 &&
          value.compare("adagrad") != 0 &&
          value.compare("adadelta") != 0 &&
          value.compare("rmsprop") != 0 &&
          value.compare("momentum") != 0) {
        printf("[Error] Unknow updater '%s' \n"
               " -updater can only be 'sgd', 'adam', 'adagrad' "
               "'rmsprop', or 'momentum' \n",
               value.c_str());
        bo = false;
      } else {
        hyper_param.updater_type = value;
      }
    } else if (list[i].compare("-k") == 0) {
      int value = atoi(list[i+1].c_str());
      if (value % 8 != 0) {
        printf("[Error] Illegal -k '%i' \n"
               " -k must be a multiple of 8 \n",
               value);
        bo = false;
      } else {
        hyper_param.num_K = value;
      }
    } else if (list[i].compare("-lr") == 0) {
      real_t value = atof(list[i+1].c_str());
      if (value <= 0) {
        printf("[Error] Illegal -lr '%f' \n"
               " -lr must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.learning_rate = value;
      }
    } else if (list[i].compare("-decay_rate") == 0) {
      real_t value = atof(list[i+1].c_str());
      if (value < 0) {
        printf("[Error] Illegal -decay_rate '%f' \n"
               " -decay_rate must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.decay_rate = value;
      }
    } else if (list[i].compare("-second_decay_rate") == 0) {
      real_t value = atof(list[i+1].c_str());
      if (value < 0) {
        printf("[Error] Illegal -second_decay_rate '%f' \n"
               " -second_decay_rate must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.second_decay_rate = value;
      }
    } else if (list[i].compare("-regu_lambda_1") == 0) {
      real_t value = atof(list[i+1].c_str());
      if (value < 0) {
        printf("[Error] Illegal -regu_lambda_1 '%f' \n"
               " -regu_lambda_1 must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.regu_lambda_1 = value;
      }
    } else if (list[i].compare("-regu_lambda_2") == 0) {
      real_t value = atof(list[i+1].c_str());
      if (value < 0) {
        printf("[Error] Illegal -regu_lambda_2 '%f' \n"
               " -regu_lambda_2 must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.regu_lambda_2 = value;
      }
    } else if (list[i].compare("-epoch") == 0) {
      int value = atoi(list[i+1].c_str());
      if (value < 0) {
        printf("[Error] Illegal -epoch '%i' \n"
               " -epoch must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.num_epoch = value;
      }
    } else if (list[i].compare("-batch_size") == 0) {
      int value = atoi(list[i+1].c_str());
      if (value < 0) {
        printf("[Error] Illegal -batch_size '%i' \n"
               " -batch_size must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.batch_size = value;
      }
    } else if (list[i].compare("-file_format") == 0) {
      std::string value = list[i+1];
      if (value.compare("libsvm") != 0 &&
          value.compare("libffm") != 0 &&
          value.compare("csv") != 0) {
        printf("[Error] Unknow file format '%s' \n"
               " -file_format can only be 'libsvm', 'libffm', "
               "or 'csv' \n",
               value.c_str());
        bo = false;
      } else {
        hyper_param.file_format = value;
      }
    } else if (list[i].compare("-cv") == 0) {
      std::string value = list[i+1];
      if (value.compare("true") != 0 &&
          value.compare("false") != 0) {
        printf("[Error] Illegal -cv '%s' \n"
               " -cv can only be 'true' or 'false' \n",
               value.c_str());
        bo = false;
      } else {
        hyper_param.cross_validation =
          value.compare("true") == 0 ? true : false;
      }
    } else if (list[i].compare("-fold") == 0) {
      int value = atoi(list[i+1].c_str());
      if (value < 0) {
        printf("[Error] Illegal -fold '%i' \n"
               " -fold must be greater than zero \n",
               value);
        bo = false;
      } else {
        hyper_param.num_folds = value;
      }
    } else if (list[i].compare("-early_stop") == 0) {
      std::string value = list[i+1];
      if (value.compare("true") != 0 &&
          value.compare("false") != 0) {
        printf("[Error] Illegal -early_stop '%s' \n"
               " -early_stop can only be 'true' or 'false' \n",
               value.c_str());
        bo = false;
      } else {
        hyper_param.early_stop =
          value.compare("true") == 0 ? true : false;
      }
    } else if (list[i].compare("-on_disk") == 0) {
      std::string value = list[i+1];
      if (value.compare("true") != 0 &&
          value.compare("false") != 0) {
        printf("[Error] Illegal -on_disk '%s' \n"
               " -on_disk can only be 'true' or 'false' \n",
               value.c_str());
        bo = false;
      } else {
        hyper_param.on_disk =
          value.compare("true") == 0 ? true : false;
      }
    } else if (list[i].compare("-infer_data") == 0) {
      printf("[Warning] -infer_data can only be used in inference. \n"
             "xLearn will ignore this option \n");
    } else if (list[i].compare("-out_data") == 0) {
      printf("[Warning] -out_data can only be used in inference. \n"
             "xLearn will ignore this option \n");
    } else { // no option match
      std::string similar_str;
      if (ss.FindSimilar(list[i], menu_, similar_str) > 7) {
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

  // Check warning
  if (hyper_param.score_func.compare("ffm") == 0 &&
      hyper_param.file_format.compare("libsvm") == 0) {
    printf("[Warning] FFM model cannot use the libsvm file. "
           "Already Change it to libffm format.\n");
    hyper_param.file_format = "libffm";
  }
  if (hyper_param.cross_validation &&
     !hyper_param.test_set_file.empty()) {
    printf("[Warning] xLearn will use cross validation. "
           "The test file will not work.\n");
  }


  if (!bo) { return false; }

  return true;
}

// Check options for inference tasks
bool Checker::check_inference_options(HyperParam& hyper_param) {
  bool bo = true;
  // Remove the first two arguments
  StringList list(args_.begin()+2, args_.end());
  StrSimilar ss;
  // Every option should have a value
  if (list.size() % 2 != 0) {
    printf("[Error] Every option should have a value \n");
    return false;
  }
  // User must set the -infer_data
  if (!ss.Find(std::string("-infer_data"), list)) {
    printf("[Error] User need to set the option [-infer_data] "
           "to specify which file storing the inference data \n");
    bo = false;
  }

  if (!bo) { return false; }

  // Check every single element
  for (int i = 0; i < list.size(); i+=2) {
    if (list[i].compare("-infer_data") == 0) {
      hyper_param.inference_file = list[i+1];
    } else if (list[i].compare("-out_data") == 0) {
      hyper_param.output_file = list[i+1];
    } else if (list[i].compare("-model_file") == 0) {
      hyper_param.model_file = list[i+1];
    } else { // options used in training
      bool bk = true;
      for (int j = 0; j < menu_.size(); ++j) {
        if (list[i].compare(menu_[j]) == 0) {
          printf("[Warning] %s can only be used in training \n"
                 "xLearn will ignore this option \n",
                 list[i].c_str());
          bk = false;
          break;
        }
      }
      if (!bk) { continue; }
      // no match
      std::string similar_str;
      if (ss.FindSimilar(list[i], menu_, similar_str) > 7) {
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
