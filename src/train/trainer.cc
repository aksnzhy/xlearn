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
//------------------------------------------------------------------------------
void Trainer::print_logo() {
  std::cout << "---------------------------------------------\n"
            << "      _\n"
            << "     | |\n"
            << "__  _| |     ___  __ _ _ __ _ __\n"
            << "\\ \\/ / |    / _ \\/ _` | '__| '_ \\ \n"
            << " >  <| |___|  __/ (_| | |  | | | |\n"
            << "/_/\\_\\_____/\\___|\\__,_|_|  |_| |_|\n\n"
            << "xLearn   -- 0.10 Version --\n"
            << "---------------------------------------------\n";
}

// Option help menu
std::string Trainer::option_help() {
  return std::string(
"--------------------------------  Train task  -------------------------------\n"
"Usage: xlearn --is_train [options] \n"
"\n"
"options: \n"
"-train_data <file_path>: path of your training data file \n"
"-test_data <file_path>: (optional, if use cross_validation) "
                         "path of your validation data file \n"
"-model_file <file_path>: path of your model checkpoint file \n"
"-score <score_function>: score_function can be 'linear', 'fm', and 'ffm' \n"
"-loss <loss_function>: loss_function can be 'squared', 'absolute', "
                        "'cross_entropy', and 'hinge' \n"
"-regular <regular>: (optional) regular term can be 'l1', 'l2' and 'l1_l2' \n"
"-updater <updater_method> : updater_method can be 'sgd', 'adagrad', 'adam' "
                             "'adadelta', 'rmsprop', and 'momentum' \n"
"-lr <learning_rate>: learning rate for optimization method \n"
"-decay_rate <decay_rate>: (optional) decay rate for optimization method \n"
"-second_decay_rate <decay_rate>: (optional) second_decay_rate for "
                                  "optimization method \n"
"-regu_lambda_1 <lambda>: (optional) lambda_1 for regular term \n"
"-regu_lambda_2 <lambda>: (optional) lambda_2 for regular term \n"
"-epoch <number>: the number of epoch \n"
"-batch_size <size>: batch size for each training \n"
"-file_format <format>: format can be 'libsvm', 'libffm', and 'csv' \n"
"--cross_validation: (optional) using cross-validation in training \n"
"-fold <number>: fold number for cross-validation \n"
"--early_stop: (optional) using early-stop in training \n"
"----------------------------------------------------------------------------\n"
"\n"
"----------------------------- Inference task -------------------------------\n"
"Usage: xlearn --is_inference [options] \n"
"-model_file <file_path>: path of your pre-trained model file \n"
"-infer_data <file_path>: path of your inference data file \n"
"-out_data <file_path>: path of your output result \n"
  );
}

// Parse and check the command line args
void Trainer::parse_and_check_options(int argc, char* argv[]) {
  // Get all of the arguments
  std::vector<std::string> args;
  for (int i = 0; i < argc, ++i) {
    args.push_back(std::string(argv[i]));
  }
  if (argc == 1) {
    throw invalid_argument(option_help());
  }
  // Parse and check args
  for (int i = 0; i < argc, ++i) {

  }
}

// Read training dataset and
void Trainer::read_problem() {

}

//------------------------------------------------------------------------------
// StartWork:
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// Finalize:
//------------------------------------------------------------------------------

} // namespace xLearn
