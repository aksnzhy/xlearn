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

#ifndef XLEARN_DIST_SOLVER_SOLVER_H_
#define XLEARN_DIST_SOLVER_SOLVER_H_

#include "src/base/common.h"
#include "src/base/thread_pool.h"
#include "src/data/hyper_parameters.h"
#include "src/data/data_structure.h"
#include "src/data/model_parameters.h"
#include "src/reader/reader.h"
#include "src/reader/parser.h"
#include "src/reader/file_splitor.h"
#include "src/score/score_function.h"
#include "src/distributed/dist_score_function.h"
#include "src/distributed/dist_loss.h"
#include "src/loss/metric.h"
#include "src/distributed/dist_checker.h"
#include "src/distributed/dist_trainer.h"
#include "src/distributed/dist_inference.h"

namespace xLearn {
//------------------------------------------------------------------------------
// Solver is entry class of xLearn, which can perform training 
// or prediction tasks. There are three important functions in this 
// class, including the Initialize(), StartWork(), and Clear() funtions.
// 
// We can use Solver class like this:
//
//  xLearn::Solver solver;
//  solver.SetTrain();   // or solver.SetPredict()
//  solver.Initialize(argc, argv);
//  solver.StartWork();
//  solver.Clear();
//------------------------------------------------------------------------------
class DistSolver {
 public:
  // Constructor and Destructor
  DistSolver() 
    : dist_score_(nullptr),
      dist_loss_(nullptr),
      metric_(nullptr) { }
  ~DistSolver() { }

  // Ser train or predict
  void SetTrain() { hyper_param_.is_train = true; }
  void SetPredict() { hyper_param_.is_train = false; }

  // Initialize the xLearn environment, including checking
  // and parsing the commad line arguments, reading problem
  // (training data or testing data), initialize model, loss, 
  // metric, and score functions, etc.
  void Initialize(int argc, char* argv[]);

  // Initialize the xLearn environment through the
  // given hyper-parameters. This function will be 
  // used for python API.
  void Initialize(HyperParam& hyper_param);

  // Start a training task or start an inference task.
  void StartWork();

  // Clear the xLearn environment.
  void Clear();

 protected:
  /* Global hyper-parameters */
  xLearn::HyperParam hyper_param_;
  /* Check the user input */
  xLearn::DistChecker dist_checker_;
  /* Global model parameters */
  xLearn::Model* model_;
  /* One Reader corresponds one data file */
  std::vector<xLearn::Reader*> reader_;
  /* Split file in cross-validation */
  xLearn::FileSpliter splitor_;
  /* linear, fm or ffm ? */
  xLearn::DistScore* dist_score_;
  /* cross-entropy or squared ? */
  xLearn::DistLoss* dist_loss_;
  /* acc, prec, recall, mae, etc */
  xLearn::Metric* metric_;
  /* ThreadPool for multi-thread training */
  ThreadPool* pool_;

  // Create object by name
  xLearn::Reader* create_reader();
  //xLearn::Score* create_score();
  xLearn::DistScore* create_score();
  xLearn::DistLoss* create_loss();
  xLearn::Metric* create_metric();

  // xLearn command line logo
  void print_logo() const;

  // Initialize function
  void init_train();
  void init_predict();
  void init_log();
  void checker(int argc, char* argv[]);
  void checker(HyperParam& hyper_param);

  // Start function
  void start_train_work();
  void start_prediction_work();

 private:
  DISALLOW_COPY_AND_ASSIGN(DistSolver);
};

} // namespace xLearn

#endif // XLEARN_DIST_SOLVER_SOLVER_H_
