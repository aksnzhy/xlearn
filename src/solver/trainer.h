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
#include "src/updater/updater.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Trainer is the core class of xLearn, which can perform standard training
// process (training set and test set) and cross_validation training process.
//------------------------------------------------------------------------------
class Trainer {
 public:
  Trainer() {}
  ~Trainer() {}

  // Init for standard training
  void Initialize(Reader* train_reader,
                  Reader* test_reader,
                  int epoch,
                  Model* model,
                  Loss* loss,
                  Metric* metric,
                  Updater* updater,
                  bool early_stop) {
    CHECK_NOTNULL(train_reader);
    CHECK_GT(epoch, 0);
    CHECK_NOTNULL(model);
    CHECK_NOTNULL(loss);
    CHECK_NOTNULL(metric);
    CHECK_NOTNULL(updater);
    train_reader_ = train_reader;
    test_reader_ = test_reader;
    epoch_ = epoch;
    model_ = model;
    loss_ = loss;
    metric_ = metric;
    updater_ = updater;
    early_stop_ = early_stop;
  }

  // Init for CV training
  void Initialize(std::vector<Reader*> reader_list,
                  int epoch,
                  Model* model,
                  Loss* loss,
                  Updater* updater,
                  bool early_stop) {
    CHECK_NE(reader_list.empty(), true);
    reader_list_.resize(reader_list.size(), NULL);
    for (int i = 0; i < reader_list.size(); ++i) {
      CHECK_NOTNULL(reader_list[i]);
      reader_list_[i] = reader_list[i];
    }
    CHECK_GT(epoch, 0);
    CHECK_NOTNULL(model);
    CHECK_NOTNULL(loss);
    CHECK_NOTNULL(updater);
    epoch_ = epoch;
    model_ = model;
    loss_ = loss;
    updater_ = updater;
    early_stop_ = early_stop;
  }

  // Standard training
  void Train();

  // cross_validation training
  void CVTrain();

  // Save model to disk file
  void SaveModel(const std::string& filename) {
    model_->Serialize(filename);
  }

 protected:
  Reader* train_reader_;
  Reader* test_reader_;
  std::vector<Reader*> reader_list_;
  int epoch_;
  Model* model_;
  Loss* loss_;
  Metric* metric_;
  Updater* updater_;
  bool early_stop_;

 private:
  DISALLOW_COPY_AND_ASSIGN(Trainer);
};

} // namespace xLearn

#endif  // XLEARN_SOLVER_TRAINER_H_
