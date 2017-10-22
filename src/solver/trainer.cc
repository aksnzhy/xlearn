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
This file is the implementation of the Trainer class.
*/

#include <stdio.h>
#include <vector>

#include "src/solver/trainer.h"
#include "src/data/data_structure.h"

namespace xLearn {

/*********************************************************
 *  Show head info                                       *
 *********************************************************/
void Trainer::show_head_info(bool validate) {
  std::cout.width(6);
  std::cout << "Epoch";
  std::cout.width(25);
  std::string str = "Train " + loss_->loss_type();
  std::cout << str;
  std::cout.width(20);
  str = "Train " + metric_->type();
  std::cout << str;
  if (validate) {
    std::cout.width(25);
    str = "Test " + loss_->loss_type();
    std::cout << str;
    std::cout.width(20);
    str = "Test " + metric_->type();
    std::cout << str;
  }
  std::cout.width(19);
  std::cout << "Time cost (s)";
  std::cout << std::endl;
}

/*********************************************************
 *  Show train info                                      *
 *********************************************************/
void Trainer::show_train_info(real_t tr_loss, const std::string& tr_metric,
                              real_t te_loss, const std::string& te_metric,
                              real_t time_cost, bool validate, index_t epoch) {
  std::cout.width(6);
  std::cout << epoch;
  std::cout.width(25);
  std::cout << std::fixed << std::setprecision(5) << tr_loss;
  std::cout.width(20);
  std::cout << std::fixed << std::setprecision(5) << tr_metric;
  if (validate) {
    std::cout.width(25);
    std::cout << std::fixed << std::setprecision(5) << te_loss;
    std::cout.width(20);
    std::cout << std::fixed << std::setprecision(5) << te_metric;
  }
  std::cout.width(19);
  std::cout << std::fixed << std::setprecision(2) << time_cost;
  std::cout << std::endl;
}

/*********************************************************
 *  Basic train function                                 *
 *********************************************************/
void Trainer::train(std::vector<Reader*> train_reader,
                    std::vector<Reader*> test_reader) {
  bool validate = test_reader.empty() ? false : true;
  // Show header info
  if (!quiet_) {
    show_head_info(validate);
  }
  for (int n = 0; n < epoch_; ++n) {
    clock_t start, end;
    start = clock();
    //----------------------------------------------------
    // Calc grad and update model
    //----------------------------------------------------
    CalcGrad_Update(train_reader);
    // we don't do any evaluation in a quiet model
    if (!quiet_) {
      //----------------------------------------------------
      // Calc Train loss
      //----------------------------------------------------
      real_t tr_loss = CalcLoss(train_reader);
      std::string tr_metric = CalcMetric(train_reader);
      //----------------------------------------------------
      // Calc Test loss
      //----------------------------------------------------
      real_t te_loss = 0.0;
      std::string te_metric;
      if (validate) {
        te_loss = CalcLoss(test_reader);
        te_metric= CalcMetric(test_reader);
      }
      end = clock();
      real_t time_cost = (real_t)(end-start) / CLOCKS_PER_SEC;
      // show train info
      show_train_info(tr_loss, tr_metric,
                      te_loss, te_metric,
                      time_cost, validate, n);
    }
  }
}

// Calculate gradient and update model
void Trainer::CalcGrad_Update(std::vector<Reader*>& reader) {
  CHECK_NE(reader.empty(), true);
  for (int i = 0; i < reader.size(); ++i) {
    reader[i]->Reset();
    DMatrix* matrix = nullptr;
    while (reader[i]->Samples(matrix)) {
      loss_->CalcGrad(matrix, *model_);
    }
  }
}

// Calculate loss value
real_t Trainer::CalcLoss(std::vector<Reader*>& reader_list) {
  CHECK_NE(reader_list.empty(), true);
  DMatrix* matrix = nullptr;
  index_t count_sample = 0;
  std::vector<real_t> pred;
  real_t loss_val = 0;
  for (int i = 0; i < reader_list.size(); ++i) {
    reader_list[i]->Reset();
    for (;;) {
      index_t tmp = reader_list[i]->Samples(matrix);
      if (tmp == 0) { break; }
      if (tmp != pred.size()) { pred.resize(tmp); }
      count_sample += tmp;
      loss_->Predict(matrix, *model_, pred);
      loss_val += loss_->Evalute(pred, matrix->Y);
    }
  }
  loss_val /= count_sample;
  return loss_val;
}

// Calculate evaluation metric
std::string Trainer::CalcMetric(std::vector<Reader*>& reader_list) {
  CHECK_NE(reader_list.empty(), true);
  DMatrix* matrix = nullptr;
  std::vector<real_t> pred;
  for (int i = 0; i < reader_list.size(); ++i) {
    reader_list[i]->Reset();
    for (;;) {
      index_t tmp = reader_list[i]->Samples(matrix);
      if (tmp == 0) { break; }
      if (tmp != pred.size()) { pred.resize(tmp); }
      loss_->Predict(matrix, *model_, pred);
      metric_->Accumulate(matrix->Y, pred);
    }
  }
  return metric_->GetMetric();
}

// The basic
void Trainer::Train() {
  // Get train Reader and test Reader
  std::vector<Reader*> tr_reader;
  tr_reader.push_back(reader_list_[0]);
  std::vector<Reader*> te_reader;
  if (reader_list_.size() == 2) {
    te_reader.push_back(reader_list_[1]);
  }
  this->train(tr_reader, te_reader);
}

// Training using cross-validation
void Trainer::CVTrain() {
  // Use the i-th reader as validation Reader
  for (int i = 0; i < reader_list_.size(); ++i) {
    printf("Cross-validation: %d/%lu: \n", i+1, reader_list_.size());
    // Get the train Reader and test Reader
    std::vector<Reader*> tr_reader;
    for (int j = 0; j < reader_list_.size(); ++j) {
      if (i == j) { continue; }
      tr_reader.push_back(reader_list_[j]);
    }
    std::vector<Reader*> te_reader;
    te_reader.push_back(reader_list_[i]);
    if (i != 0) {
      // Re-init current model parameters
      model_->Reset();
    }
    this->train(tr_reader, te_reader);
  }
}

} // namespace xLearn
