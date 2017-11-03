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
//------------------------------------------------------------------------------
// Parameters for optimization method
//------------------------------------------------------------------------------
  /* Learning rate */
  real_t learning_rate = 0.2;
  /* lambda for regularize. xLearn uses L2-regular */
  real_t regu_lambda = 0.00002;
  /* Hyper param for init model parameters */
  real_t model_scale = 0.66;
  /* Number of epoch. 
  This value could be changed in early-stop */
  int num_epoch = 10;
  /* Size (MB) of memory buffer for on-disk training */
  int working_set = 200;
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
  We must set this value in training task. */
  std::string train_set_file;
  /* Filename of test dataset 
  We must set this value in predication task. */
  std::string test_set_file;
  /* Filename for validation set
  This value can be empty. */
  std::string validate_set_file;
  /* Filename of model checkpoint
  model_file = train_set_file + ".model" */
  std::string model_file;
  /* Filename of output result for prediction
  output_file = test_set_file + ".out" */
  std::string output_file;
  /* Filename of log file */
  std::string log_file = "/tmp/xlearn_log";
//------------------------------------------------------------------------------
// Parameters for validation
//------------------------------------------------------------------------------
  /* True for using cross-validation and
  False for not */
  bool cross_validation = false;
  /* Number of folds in cross-validation */
  int num_folds = 5;
  /* True for using early-stop and
  False for not */
  bool early_stop = true;

  // Check and fix the conflict of hyper-parameters
  bool CheckConflict(std::string& err_info) {
    err_info.clear();
    bool bo = true;
    // Confict for on-disk training
    if (this->on_disk) {
      if (this->cross_validation) {
        err_info += "[Warning] On-disk training doesn't support "
                    "cross-validation and xLearn will disable it. \n";
        this->cross_validation = false;
        bo = false;
      }
    }
    // Conflict for cross-validation
    if (this->cross_validation) {
      if (this->early_stop) {
        err_info += "[Warning] cross-validation doesn't support "
                    "early-stopping and xLearn will disable it. \n";
        this->early_stop = false;
        bo = false;
      }
      if (!this->validate_set_file.empty()) {
        err_info += "[Warning] xLearn has already been set to use "
                    "cross-validation and will ignore the validation file. \n";
        this->validate_set_file.clear();
        bo = false;
      }
      if (this->quiet) {
        err_info += "[Warning] Quiet training cannot be used under "
                    "cross-validation. \n";
        this->quiet = false;
        bo = false;
      }
    }
    // Conflict for early-stop
    if (this->early_stop) {
      if (this->validate_set_file.empty()) {
        err_info += "[Warning] The validation file cannot be empty when "
                    "setting early-stopping. \n";
        this->early_stop = false;
        bo = false;
      }
    }
    // Conflict for metric
    if (this->loss_func.compare("cross-entropy") == 0) {
      if (this->metric.compare("mae") == 0 ||
          this->metric.compare("rmsd") == 0 ||
          this->metric.compare("mape") == 0) {
        err_info += "[Warning] The " + this->metric + " can only be used "
                    "in regression tasks. Change it to -x acc .\n";
        this->metric = "acc";
        bo = false;
      }
    }
    if (this->loss_func.compare("squared") == 0) {
      if (this->metric.compare("acc") == 0 ||
          this->metric.compare("prec") == 0 ||
          this->metric.compare("recall") == 0 ||
          this->metric.compare("f1") == 0) {
        err_info += "[Warning] The " + this->metric + " can only be used "
                    "in classification tasks. Change it to -x mae .\n";
        this->metric = "mae";
        bo = false;
      }
    }
    return bo;
  }
};

}  // namespace XLEARN

#endif  // XLEARN_DATA_HYPER_PARAMETER_H_
