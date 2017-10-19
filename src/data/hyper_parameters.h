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
// Parameters for current task
//------------------------------------------------------------------------------
  /* Train or Predict
  True for train, false for predict */
  bool is_train = true;
  /* On-disk for limited memory
  True for on-disk training, false for
  in-memory training */
  bool on_disk = false;
  /* Don't print any evaluation information
  during the training. Just train the model */
  bool quiet = false;
  /* Score function
  Could be 'linear', 'fm', or 'ffm' */
  std::string score_func = "linear";
  /* Loss function
  Could be 'cross-entropy', 'squared', or 'hinge' */
  std::string loss_func = "corss-entropy";
  /* metric function
  Could be acc', 'prec', 'recall', 'roc',
  'auc', 'mae', or 'mse' */
  std::string metric = "acc";
//------------------------------------------------------------------------------
// Parameters for optimization method
//------------------------------------------------------------------------------
  /* Learning rate */
  real_t learning_rate = 0.2;
  /* lambda for regularizer
  xLearn uses sparse regularizer */
  real_t regu_lambda = 0.00002;
  /* Number of epoch
  Could be changed in early-stop */
  int num_epoch = 5;
  /* Sample size for each training iteration
  reader->Samples(matrix) will return this
  value to user */
  int sample_size = 200;
  /* True for use instance-wise
  normalization. False for not */
  bool norm = true;
//------------------------------------------------------------------------------
// Parameters for dataset
//------------------------------------------------------------------------------
  /* Number of feature
  Include the bias term '0'
  We get this value from initialization */
  index_t num_feature = 0;
  /* Number of model parameters
  We get this value from initialization */
  index_t num_param = 0;
  /* Number of lateny factor for fm and ffm
  We get this value from initialization */
  index_t num_K = 4;
  /* Number of field, used by ffm tasks
  We get this value from initialization */
  index_t num_field = 0;
  /* Filename of training set
  We must set this value in training */
  std::string train_set_file;
  /* Filename for test set
  This value can be empty */
  std::string test_set_file;
  /* Filename of prediction set
  We must set this value in prediction */
  std::string predict_file;
  /* Filename of model checkpoint */
  std::string model_file = "./xlearn_model";
  /* Filename of output result of prediction */
  std::string output_file = "./xlearn_out";
  /* Filename of log file */
  std::string log_file = "/tmp/xlearn_log/";
//------------------------------------------------------------------------------
// Parameters for validation
//------------------------------------------------------------------------------
  /* Use cross-validation or not */
  bool cross_validation = false;
  /* Number of folds in cross-validation */
  int num_folds = 5;
  /* Use early-stop or not */
  bool early_stop = false;
};

}  // namespace XLEARN

#endif  // XLEARN_DATA_HYPER_PARAMETER_H_
