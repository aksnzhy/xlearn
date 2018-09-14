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
This file defines the Trainer class.
*/

#ifndef XLEARN_SOLVER_TRAINER_H_
#define XLEARN_SOLVER_TRAINER_H_

#include <vector>

#include "src/base/common.h"
#include "src/base/format_print.h"
#include "src/reader/reader.h"
#include "src/data/model_parameters.h"
#include "src/loss/loss.h"
#include "src/loss/metric.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Trainer is the core class of xLearn, which can perform
// standard training process (training set and test set), as 
// well as the cross-validation training process.
//------------------------------------------------------------------------------
class Trainer {
 public:
  // Constructor and Destructor
  Trainer() {}
  ~Trainer() {}

  // Invoke this function before we use this class
  void Initialize(std::vector<Reader*>& reader_list,
                  int epoch,
                  Model* model,
                  Loss* loss,
                  Metric* metric,
                  bool early_stop,
                  int stop_window,
                  bool quiet) {
    CHECK_NE(reader_list.empty(), true);
    CHECK_GT(epoch, 0);
    CHECK_GT(stop_window, 0);
    CHECK_NOTNULL(model);
    CHECK_NOTNULL(loss);
    // Do not check metric == nullptr
    reader_list_ = reader_list;
    epoch_ = epoch;
    model_ = model;
    loss_ = loss;
    metric_ = metric;
    early_stop_ = early_stop;
    stop_window_ = stop_window;
    quiet_ = quiet;
  }

  // Training without cross-validation
  void Train();

  // Training using cross-validation
  void CVTrain();

  // Save model to disk file
  void SaveModel(const std::string& filename) {
    CHECK_NE(filename.empty(), true);
    CHECK_NE(filename.compare("none"), 0);
    model_->Serialize(filename);
  }

  // Save txt model to disk file
  void SaveTxtModel(const std::string& filename) {
    CHECK_NE(filename.empty(), true);
    CHECK_NE(filename.compare("none"), 0);
    model_->SerializeToTXT(filename);
  }

 protected:
  /* The reader_list_ contains both of the 
  training data and the validation data. */
  std::vector<Reader*> reader_list_;
  /* Number of epoch */
  int epoch_;
  /* Using early-stopping ? */
  bool early_stop_;
  /* Window size for early stopping */
  int stop_window_;
  /* quiet training ? */
  bool quiet_;
  /* Model parameter */
  Model* model_;
  /* Loss function */
  Loss* loss_;
  /* Evaluation metric */
  Metric* metric_;
  /* Store each metric info of cross-validation */
  std::vector<MetricInfo> metric_info_;

  // Basic train function
  void train(std::vector<Reader*>& train_reader,
             std::vector<Reader*>& test_reader);

  // Caculate gradient and update model.
  // Return training loss.
  real_t calc_gradient(std::vector<Reader*>& reader_list);

  // Calculate loss value and evaluation metric.
  MetricInfo calc_metric(std::vector<Reader*>& reader_list);

  // Calculate average metric for cross-validation
  void show_average_metric();

  // Print information during the training.
  void show_head_info(bool validate);
  void show_train_info(real_t tr_loss, 
                       real_t te_loss,
                       real_t te_metric,
                       real_t time_cost, 
                       bool validate,
                       index_t epoch);

 private:
  DISALLOW_COPY_AND_ASSIGN(Trainer);
};

} // namespace xLearn

#endif  // XLEARN_SOLVER_TRAINER_H_
