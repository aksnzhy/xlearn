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

This file defines the hyper-parameters used by f2m.
*/

#ifndef F2M_DATA_HYPER_PARAMETER_H_
#define F2M_DATA_HYPER_PARAMETER_H_

#include <string>

#include "src/data/data_structure.h"

namespace f2m {

//------------------------------------------------------------------------------
// We use a single data structure - HyperParam to handle all the
// hyper parameters used by f2m.
//------------------------------------------------------------------------------
struct HyperParam {
  // Train or Predict?
  bool is_train = true;
  // binary_classification or regression ?
  std::string task_type = "binary_classification";
  // linear, 
  std::string model_type = "linear";
  // Store gradient in a sparse or dense way.
  bool is_sparse = false;
  // Control the learning step.
  real_t learning_rate = 0.01;
  // Parser
  ParserType parser = LibSVM;
  // The index of the max feature (including the bias).
  index_t max_feature = 0;
  // Number of model parameters.
  index_t num_param = 0;
  // (optional) The number of field, only used by ffm.
  int num_field = 0;
  // The number of latent factors, only used by fm and ffm.
  int num_factor = 0;
  // Indicate which Updater we use in current task.
  UpdaterType updater = SGD;
  // The decay factor used by Updater.
  real_t decay_rate = 0.9;
  // The sceond decay factor used by Updater.
  real_t second_decay_rate = 0.9;
  // lambda for regularizer
  real_t regu_lambda = 0.01;
  // Indicate which regularizer we use in current task.
  RegularType regu_type = L2;
  // Filename of trainning data set.
  std::string train_set_file;
  // Filename of test data set.
  std::string test_set_file;
  // File for saving model checkpoint.
  std::string model_checkpoint_file;
  // Number of iteration.
  int num_iteration = 100;
  // If using cross validation
  bool cross_validation = true;
  // Number of folds for cross-validation.
  int num_folds = 5;
  // in-memory or on-disk trainning.
  bool in_memory_trainning = true;
  // Mini-batch size in each iteration..
  int batch_size = 0;
  // Using Early-stop ?
  bool early_stop = false;
  // Using sigmoid ?
  bool sigmoid = false;
};

} // namespace f2m

#endif // F2M_DATA_HYPER_PARAMETER_H_
