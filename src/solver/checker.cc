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

#include "src/solver/checker.h"

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
"    -model_file <path> (optional, use '/tmp/xlearn-model' by default): \n"
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
"----------------------------------------------------------------------------\n"
"\n"
"----------------------------- Inference task -------------------------------\n"
"Usage: xlearn --is_inference [options] \n"
"\n"
"options: \n\n"
/* must have value */
"   -model_file <path>: path of your pre-trained model file \n"
/* must have value */
"   -infer_data <path>: path of your inference data file \n"
/* must have value */
"   -out_data <path>: path of your output file \n"
"----------------------------------------------------------------------------\n"
  );
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
}

// Check and parse user input
bool Checker::Check(HyperParam& hyper_param) {
  // Do not have any args
  if (args_.size() == 1) {
    std::cout << option_help();
    return false;
  }
  // Parse and check argument
  if (args_[1].compare("--is_train") == 0) {
    hyper_param.is_train = true;
    return check_train_options(hyper_param);
  } else if (args_[1].compare("--is_inference") == 0) {
    hyper_param.is_train = false;
    return check_inference_options(hyper_param);
  } else {
    std::cout << "Arguments error. Please use: \n"
              << " 'xlearn --is_train [options]' for training task \n"
              << "or \n 'xlearn --is_inference [options]' for inference task\n"
              << "Type 'xlearn' for the details of the [options]\n";
    return false;
  }
}


// Check options for training tasks
bool Checker::check_train_options(HyperParam& hyper_param) {
  /*
  bool bo = true;
  // Check the arguments that must be setted by user
  StringList list(args.begin()+2, args.end());
  // user must set the -train_data
  if (!find_str(list, std::string("-train_data"))) {
    std::cout << "User need to set the option [-train_data] "
              << "to specify the input training data file.\n";
    bo = false;
  }
  // user must set the -score
  if (!find_str(list, std::string("-score"))) {
    std::cout << "User need to set the option [-score] "
              << "to specify the score function, which can be 'linear', "
              << "'fm', and 'ffm' \n";
    bo = false;
  }
  // user must set the -loss
  if (!find_str(list, std::string("-loss"))) {
    std::cout << "User need to set the option [-loss] "
              << "to specify the loss function, which can be 'squared', "
              << "'absolute', 'cross_entropy', and 'hinge' \n";
    bo = false;
  }
  if (!bo) { return false; }
  // Check every single elements
*/
  return true;
}

// Check options for inference tasks
bool Checker::check_inference_options(HyperParam& hyper_param) {
  return true;
}

} // namespace xLearn
