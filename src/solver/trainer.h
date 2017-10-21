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

This file defines the Trainer class.
*/

#ifndef XLEARN_SOLVER_TRAINER_H_
#define XLEARN_SOLVER_TRAINER_H_

#include <vector>

#include "src/base/common.h"
#include "src/reader/reader.h"
#include "src/data/model_parameters.h"
#include "src/loss/loss.h"
#include "src/loss/metric.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Trainer is the core class of xLearn, which can perform standard training
// process (training set and test set) and cross_validation training process.
//------------------------------------------------------------------------------
class Trainer {
 public:
  Trainer() {}
  ~Trainer() {}

  // Invoke this function before we use this class
  void Initialize(std::vector<Reader*>& reader_list,
                  int epoch,
                  Model* model,
                  Loss* loss,
                  Metric* metric,
                  bool early_stop,
                  bool quiet) {
    CHECK_NE(reader_list.empty(), true);
    CHECK_GT(epoch, 0);
    CHECK_NOTNULL(model);
    CHECK_NOTNULL(loss);
    CHECK_NOTNULL(metric);
    reader_list_ = reader_list;
    epoch_ = epoch;
    model_ = model;
    loss_ = loss;
    metric_ = metric;
    early_stop_ = early_stop;
    quiet_ = quiet;
  }

  // Training without cross-validation
  void Train();

  // Training using cross-validation
  void CVTrain();

  // Save model to disk file
  void SaveModel(const std::string& filename) {
    CHECK_NE(filename.compare("none"), 0);
    model_->Serialize(filename);
  }

 protected:
  std::vector<Reader*> reader_list_;
  int epoch_;
  Model* model_;
  Loss* loss_;
  Metric* metric_;
  bool early_stop_;
  bool quiet_;

  // Basic train function
  void train(std::vector<Reader*> train_reader,
             std::vector<Reader*> test_reader);

  void show_head_info(bool validate);
  void show_train_info(real_t tr_loss, const std::string& tr_metric,
                       real_t te_loss, const std::string& te_metric,
                       real_t time_cost, bool validate, index_t n);

  // Caculate gradient and update model
  void CalcGrad_Update(std::vector<Reader*>& reader_list);
  // Calculate loss value
  real_t CalcLoss(std::vector<Reader*>& reader_list);
  // Calculate evaluation metric
  std::string CalcMetric(std::vector<Reader*>& reader_list);

 private:
  DISALLOW_COPY_AND_ASSIGN(Trainer);
};

} // namespace xLearn

#endif  // XLEARN_SOLVER_TRAINER_H_
