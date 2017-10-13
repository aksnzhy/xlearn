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

This file defines the hyper-parameters used by xLearn.
*/

#ifndef XLEARN_DATA_HYPER_PARAMETER_H_
#define XLEARN_DATA_HYPER_PARAMETER_H_

#include <string>

#include "src/data/data_structure.h"

namespace xLearn {

//------------------------------------------------------------------------------
// We use a single data structure - HyperParam to handle all the
// hyper parameters used by xLearn.
//------------------------------------------------------------------------------
struct HyperParam {
//------------------------------------------------------------------------------
// Parameters for current task.
//------------------------------------------------------------------------------
  /* Train or Predict */
  bool is_train = true;                     // True for train, false for predict
  /* On-disk for limited memory */
  bool on_disk = false;                     // True for on-disk training, false
                                            // for in-memory training
  /* Score function */
  std::string score_func = "fm";            // 'linear', 'fm', or 'ffm'
  /* Loss function */
  std::string loss_func = "corss-entropy";  // 'cross-entropy', 'squared',
                                            // 'absolute', or 'hinge'
//------------------------------------------------------------------------------
// Parameters for optimization method.
//------------------------------------------------------------------------------
  /* Learning rate */
  real_t learning_rate = 0.002;        // Control the learning setp
  /* Update function */
  std::string updater_type = "sgd";    // 'sgd', 'adagrad', 'adadelta', 'adam',
                                       // 'rmsprop', 'nesterov', or 'momentum'
  /* decay_rate for updater */
  real_t decay_rate = 0.5;             // The decay factor used by updater
  /* lambda for regularizer */
  real_t regu_lambda = 0.00002;        // xLearn uses sparse regularizer
  /* Number of epoch */
  int num_epoch = 15;                  // Could be changed in early-stop
  /* Sample size for on-disk training */
  int sample_size = 200;               // reader->Samples(matrix) will
                                       // return this value to user
//------------------------------------------------------------------------------
// Parameters for dataset
//------------------------------------------------------------------------------
  /* Number of feature */
  index_t num_feature = 0;              // Include the bias term '0'
  /* Number of model parameters */
  index_t num_param = 0;                // Need to be calculated in init()
                                        // function in the Solver class
  /* Lateny factor for fm and ffm */
  index_t num_K = 8;                    // This value must be a multiple of 8,
                                        // for example, 8, 16, 32 ...
  /* Number of field */
  index_t num_field = 0;                // Field id will start from 0
  /* Filename for training set */
  std::string train_set_file;           // Must have value in training task
  /* Filename for test set */
  std::string test_set_file;            // This value can be empty
  /* Filename for prediction set */
  std::string predict_file;             // Must have value in predict task
  /* Filename of model */
  std::string model_file = "./xlearn_model";
  /* Filename of output result */
  std::string output_file = "./xlearn_out";
  /* Filename of log file */
  std::string log_file = "./xlearn_log";  // This is not the final filename
//------------------------------------------------------------------------------
// Parameters for validation
//------------------------------------------------------------------------------
  /* Use cv or not */
  bool cross_validation = false;  // True for using cv, false for not
  /* Number of folds in cv */
  int num_folds = 5;
  /* Use early-stop or not */
  bool early_stop = false;        // True for using early-stop, false for not
};

}  // namespace XLEARN

#endif  // XLEARN_DATA_HYPER_PARAMETER_H_
