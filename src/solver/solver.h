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
of the xLearn.
*/

#ifndef XLEARN_SOLVER_SOLVER_H_
#define XLEARN_SOLVER_SOLVER_H_

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/data/data_structure.h"
#include "src/data/model_parameters.h"
#include "src/reader/reader.h"
#include "src/reader/parser.h"
#include "src/reader/file_splitor.h"
#include "src/updater/updater.h"
#include "src/score/score_function.h"
#include "src/loss/loss.h"
#include "src/solver/checker.h"

namespace xLearn {
//------------------------------------------------------------------------------
// Solver is entry class of xLearn, which can perform training or inference
// tasks. There are three important functions in this class, including the
// Initialize(), StartWork(), and Finalize() funtions.
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
  Checker checker_;
  std::vector<Reader*> reader_;
  FileSpliter splitor_;
  Parser* parser_;
  Model* model_;
  Updater* updater_;
  Score* score_;
  Loss* loss_;

  Parser* create_parser();
  Reader* create_reader();
  Updater* create_updater();
  Score* create_score();
  Loss* create_loss();
  void start_train_work();
  void start_inference_work();
  void finalize_train_work();
  void finalize_inference_work();
  index_t find_max_feature(DMatrix* matrix, int num_samples);
  index_t find_max_field(DMatrix* matrix, int num_samples);

  void print_logo() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(Solver);
};

} // namespace xLearn

#endif // XLEARN_SOLVER_SOLVER_H_
