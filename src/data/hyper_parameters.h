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

This file defines the basic hyper-parameters used by xLearn.
*/

#ifndef XLEARN_DATA_HYPER_PARAMETER_H_
#define XLEARN_DATA_HYPER_PARAMETER_H_

#include <string>

#include "src/data/data_structure.h"

namespace xLearn {

//------------------------------------------------------------------------------
// We use a single data structure - HyperParam to handle all of
// the hyper parameters used by xLearn.
//------------------------------------------------------------------------------
struct HyperParam {
//------------------------------------------------------------------------------
// Baisc parameters for current task
//------------------------------------------------------------------------------
  /* Train or Predict
  True for train, and false for predict */
  bool is_train = true;
  /* On-disk training for limited memory
  True for on-disk training, and false for
  in-memory training */
  bool on_disk = false;
  /* Don't print any evaluation information during
  the training, and just train the model */
  bool quiet = false;
  /* Score function. For now, it could
  be 'linear', 'fm', or 'ffm' */
  std::string score_func = "linear";
  /* Loss function. For now, it could
  be 'cross-entropy', 'squared', or 'hinge' */
  std::string loss_func = "cross-entropy";
  /* Metric function. For now, it could
  be 'acc', 'prec', 'recall', 'f1', 'auc',
  'mae', or 'mape' */
  std::string metric = "acc";
//------------------------------------------------------------------------------
// Parameters for optimization method
//------------------------------------------------------------------------------
  /* Learning rate */
  real_t learning_rate = 0.2;
  /* lambda for regularize. xLearn uses L2-regular */
  real_t regu_lambda = 0.00002;
  /* Hyper param for init model parameters */
  real_t model_scale = 0.66;
  /* Number of epoch. This value could
  be changed in early-stop */
  int num_epoch = 10;
  /* Sample size for each training iterator, and
  reader->Samples(matrix) will return this
  value to user in on-disk training, while in in-memory
  training, xlearn always samples the whole data set
  at each epoch */
  int sample_size = 20000;
  /* True for using instance-wise
  normalization, and False for not */
  bool norm = true;
//------------------------------------------------------------------------------
// Parameters for dataset
//------------------------------------------------------------------------------
  /* Number of feature */
  index_t num_feature = 0;
  /* Number of total model parameters */
  index_t num_param = 0;
  /* Number of lateny factor for fm and ffm */
  index_t num_K = 4;
  /* Number of field, used by ffm tasks */
  index_t num_field = 0;
  /* Filename of training dataset
  We must set this value in training task */
  std::string train_set_file;
  /* Filename for test set
  This value can be empty */
  std::string test_set_file;
  /* Filename of prediction set
  We must set this value in prediction astk */
  std::string predict_file;
  /* Filename of model checkpoint */
  std::string model_file = "./xlearn_model";
  /* Filename of output result for prediction */
  std::string output_file = "./xlearn_out";
  /* Filename of log file */
  std::string log_file = "./xlearn_log";
//------------------------------------------------------------------------------
// Parameters for validation
//------------------------------------------------------------------------------
  /* True for using cross-validation, and
  false for not */
  bool cross_validation = false;
  /* Number of folds in cross-validation */
  int num_folds = 5;
  /* True for using early-stop, and
  False for not */
  bool early_stop = false;
};

}  // namespace XLEARN

#endif  // XLEARN_DATA_HYPER_PARAMETER_H_
