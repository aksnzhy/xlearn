//------------------------------------------------------------------------------
// Copyright (c) 2018 by contributors. All Rights Reserved.
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
  /* Train or Predict.
  True for train, and false for predict. */
  bool is_train = true;
  /* On-disk training for limited memory.
  True for on-disk training, and false for in-memory training. */
  bool on_disk = false;
  /* Don't print any evaluation information 
  during the training, and just train the model.
  Setting this option to true will accerlate the training. */
  bool quiet = false;
  /* Score function. 
  For now, it can be 'linear', 'fm', or 'ffm' */
  std::string score_func = "linear";
  /* Loss function. 
  For now, it can be 'cross-entropy' and 'squared' */
  std::string loss_func = "cross-entropy";
  /* Metric function. 
  For now, it can be 'acc', 'prec', 'recall', 
  'f1', 'mae', 'rmsd', 'mape', or 'none' */
  std::string metric = "none";
  /* Number of thread existing in the thread pool */
  int thread_number = 0;
//------------------------------------------------------------------------------
// Parameters for optimization method
//------------------------------------------------------------------------------
  /* Optimization method */
  std::string opt_type = "adagrad";
  /* auxiliary size for gradient cache */
  index_t auxiliary_size = 2;
  /* Learning rate */
  real_t learning_rate = 0.2;
  /* lambda for regularize. xLearn uses L2-regular */
  real_t regu_lambda = 0.00002;
  /* Hyper param for init model parameters */
  real_t model_scale = 0.66;
  /* Used for ftrl */
  real_t alpha = 0.3;
  real_t beta = 1.0;
  real_t lambda_1 = 0.00001;
  real_t lambda_2 = 0.00002;
  /* Number of epoch. 
  This value could be changed in early-stop */
  int num_epoch = 10;
  /* True for using instance-wise 
  normalization, and False for not */
  bool norm = true;
  /* Using lock-free AdaGard to accelerate training */
  bool lock_free = true;
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
  We must set this value in training task. */
  std::string train_set_file;
  /* Filename of test dataset 
  We must set this value in predication task. */
  std::string test_set_file;
  /* Filename for validation set
  This value can be empty. */
  std::string validate_set_file;
  /* Filename of model checkpoint
  On default, model_file = train_set_file + ".model" */
  std::string model_file;
  /* Pre-trained model for online learning */
  std::string pre_model_file;
  /* Filename of the txt model checkpoint 
  On default, txt_model_file = none */
  std::string txt_model_file = "none";
  /* Filename of output result for prediction
  output_file = test_set_file + ".out" */
  std::string output_file;
  /* Filename of log file */
  std::string log_file = "/tmp/xlearn_log";
  /* Block size for on-disk training */
  int block_size = 500;  // 500 MB
//------------------------------------------------------------------------------
// Parameters for validation
//------------------------------------------------------------------------------
  /* True for using cross-validation and
  False for not */
  bool cross_validation = false;
  /* Number of folds in cross-validation */
  int num_folds = 3;
  /* True for using early-stop and
  False for not */
  bool early_stop = true;
  /* Early stop window size */
  int stop_window = 2;
  /* Convert predition output to 0 and 1 */
  bool sign = false;
  /* Convert predition output using sigmoid */
  bool sigmoid = false;
//------------------------------------------------------------------------------
// Parameters for distributed learning
//------------------------------------------------------------------------------
  /* Batch size for gradient descent */
  int batch_size = 1000000;
  /* Number of worker for compute gradient */
  int num_worker = 0;
  /* Number of parameter server for store model parameters */
  int num_server = 0;
};

}  // namespace XLEARN

#endif  // XLEARN_DATA_HYPER_PARAMETER_H_
