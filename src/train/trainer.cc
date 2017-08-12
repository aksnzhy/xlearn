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
This file is the implementation of the Trainer class.
*/

#include "src/train/trainer.h"

#include <vector>
#include <string>
#include <stdexcept>

namespace xLearn {

//------------------------------------------------------------------------------
// Initialize:
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//         _
//        | |
//   __  _| |     ___  __ _ _ __ _ __
//   \ \/ / |    / _ \/ _` | '__| '_ \
//    >  <| |___|  __/ (_| | |  | | | |
//   /_/\_\______\___|\__,_|_|  |_| |_|
//
//      xLearn   -- 0.10 Version --
//------------------------------------------------------------------------------
void Trainer::print_logo() {
  std::cout <<
"----------------------------------------------------------------------------\n"
            << "      _\n"
            << "     | |\n"
            << "__  _| |     ___  __ _ _ __ _ __\n"
            << "\\ \\/ / |    / _ \\/ _` | '__| '_ \\ \n"
            << " >  <| |___|  __/ (_| | |  | | | |\n"
            << "/_/\\_\\_____/\\___|\\__,_|_|  |_| |_|\n\n"
            << "   xLearn   -- 0.10 Version --\n"
            <<
"----------------------------------------------------------------------------\n";
}

// Option help menu
std::string Trainer::option_help() {
  return std::string(
"--------------------------------  Train task  ------------------------------\n"
"Usage: xlearn --is_train [options] \n"
"\n"
"options: \n\n"
/* must have value */
"    -train_data <path>: path of your training data file \n"
"    -test_data <path> (optional, if use cross_validation): \n"
"                      path of your validation data file \n"
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

// Parse and check the command line args
void Trainer::parse_and_check_options(int argc, char* argv[]) {
  // Get all of the arguments
  std::vector<std::string> args;
  for (int i = 0; i < argc; ++i) {
    args.push_back(std::string(argv[i]));
  }
  if (argc == 1) { // do not have any args
    std::cout << option_help();
    exit(0);
  }
  // Parse and check arguments
  if (args[1].compare("--is_train") == 0) {
    check_train_options(argc, argv);
  } else if (args[1].compare("--is_inference") == 0) {
    check_inference_options(argc, argv);
  } else {
    LOG(FATAL) << "Arguments error! Please use: \n"
               << "'xlearn --is_train [options]' for training task \n"
               << "or 'xlearn --is_inference [options]' for inference task\n";
  }
  // Print logo
  print_logo();
}

// Parse and check options for training task
void Trainer::check_train_options(int argc, char* argv[]) {

}

// Parse and check options for inference task
void Trainer::check_inference_options(int argc, char* argv[]) {

}

// Read training dataset and
void Trainer::read_problem(std::string train_set_file,
                           std::string test_set_file) {

}

// Initialize Trainer
void Trainer::Initialize(int argc, char* argv[]) {
  // Parse and check arguments
  parse_and_check_options(argc, argv);
}

//------------------------------------------------------------------------------
// StartWork:
//------------------------------------------------------------------------------

void Trainer::StartWork() {}

//------------------------------------------------------------------------------
// Finalize:
//------------------------------------------------------------------------------

void Trainer::Finalize() {}

} // namespace xLearn
