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
#include "src/score/score_function.h"
#include "src/loss/loss.h"
#include "src/loss/metric.h"
#include "src/solver/checker.h"
#include "src/solver/trainer.h"

namespace xLearn {
//------------------------------------------------------------------------------
// Solver is entry class of xLearn, which can perform training or inference
// tasks. There are three important functions in this class, including the
// Initialize(), StartWork(), and Finalize() funtions.
// We can use Solver class like this:
//
//  xLearn::Solver solver;
//  solver.SetPredict();   // or solver.SetPredict()
//  solver.Initialize(argc, argv);
//  solver.StartWork();
//  solver.Finalize();
//------------------------------------------------------------------------------
class Solver {
 public:
  // Constructor and Desstructor
  Solver() { }
  ~Solver() { }

  // Ser train or predict
  void SetTrain() { hyper_param_.is_train = true; }
  void SetPredict() { hyper_param_.is_train = false; }

  // Initialize the xLearn environment, including checking
  // and parsing the commad line arguments, reading problem
  // (training data or testing data), and create model parameters
  void Initialize(int argc, char* argv[]);

  // Start training task or start inference task
  void StartWork();

  // Finalize the xLearn environment
  void FinalizeWork();

 protected:
  // Main class used by Solver
  xLearn::HyperParam hyper_param_;
  xLearn::Checker checker_;
  xLearn::Model *model_;
  std::vector<xLearn::Reader*> reader_;
  xLearn::FileSpliter splitor_;
  xLearn::Score* score_;
  xLearn::Loss* loss_;
  xLearn::Metric* metric_;

  // Create ovject by name
  xLearn::Reader* create_reader();
  xLearn::Score* create_score();
  xLearn::Loss* create_loss();
  xLearn::Metric* create_metric();

  // Initialize function
  void init_train();
  void init_predict();

  // Used by start function
  void start_train_work();
  void start_inference_work();

  // Used by finalize funcrion
  void finalize_train_work();
  void finalize_inference_work();

  // Read problem and set feature and field
  index_t find_max_feature(DMatrix* matrix, int num_samples);
  index_t find_max_field(DMatrix* matrix, int num_samples);

  // Used by log file suffix
  std::string get_host_name();
  std::string get_user_name();
  std::string print_current_time();
  std::string get_log_file();

  // xLearn command line logo
  void print_logo() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(Solver);
};

} // namespace xLearn

#endif // XLEARN_SOLVER_SOLVER_H_
