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

This file defines the Solver class, which is the entry
class of the xLearn.
*/

#ifndef XLEARN_TRAIN_SOLVER_H_
#define XLEARN_TRAIN_SOLVER_H_

#include <vector>
#include <string>

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"

namespace xLearn {

typedef std::vector<std::string> StringList;

//------------------------------------------------------------------------------
// Solver is entry class of xLearn, which can perform training or inference
// tasks. There are three important functions in this class, including
// Initialize(), StartWork(), and Finalize() .
//------------------------------------------------------------------------------
class Solver {
 public:
  // Constructor and Desstructor
  Solver() { }
  ~Solver() { }

  // Initialize the xLearn environment, including checking
  // and parsing the arguments, reading problem (training data
  // or testing data), create model parameters, and so on.
  void Initialize(int argc, char* argv[]);

  // Start training task or start inference task.
  void StartWork();

  // Finalize the xLearn environment.
  void Finalize();

 protected:
  HyperParam hyper_param_;

  void print_logo();
  void start_train();
  void start_inference();
  std::string option_help();
  void parse_and_check_options(int argc, char* argv[]);
  void check_train_options(StringList& args);
  void check_inference_options(StringList& args);
  void read_problem(std::string train_set_file,
                    std::string test_set_file);

 private:
  DISALLOW_COPY_AND_ASSIGN(Solver);
};

} // namespace xLearn

#endif // XLEARN_TRAIN_SOLVER_H_
