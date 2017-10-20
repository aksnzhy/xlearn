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

// Calculate gradient and update model
void Trainer::CalcGrad_Update() {
  train_reader_->Reset();
  DMatrix* matrix = nullptr;
  while (train_reader_->Samples(matrix)) {
    loss_->CalcGrad(matrix, *model_);
  }
}

// Calculate loss
void Trainer::CalcLoss_Metric(Reader* reader,
                              real_t* loss,
                              real_t* metric) {
  reader->Reset();
  DMatrix* matrix = nullptr;
  index_t count_sample = 0;
  real_t loss_val = 0.0;
  index_t real_pos_example = 0;
  index_t real_neg_example = 0;
  index_t pre_pos_example = 0;
  index_t pre_neg_example = 0;
  static std::vector<real_t> pred;
  int tmp = 0;
  while (1) {
    tmp = reader->Samples(matrix);
    if (tmp == 0) { break; }
    if (tmp != pred.size()) { pred.resize(tmp); }
    count_sample += tmp;
    loss_->Predict(matrix, *model_, pred);
    metric_->Accumulate(&real_pos_example,
                      &real_neg_example,
                      &pre_pos_example,
                      &pre_neg_example,
                      matrix->Y,
                      pred);
    loss_val += loss_->Evalute(pred, matrix->Y);
  }
  loss_val /= count_sample;
  *loss = loss_val;
  metric_->Set(real_pos_example, real_neg_example,
               pre_pos_example, pre_neg_example);
  *metric = metric_->GetMetric();
}

/*********************************************************
 *  Show head info                                       *
 *********************************************************/
void Trainer::show_head_info(bool validate) {
  std::cout.width(6);
  std::cout << "Epoch";
  std::cout.width(15);
  std::cout << "Train loss";
  std::cout.width(20);
  std::string str = "Train " + metric_->type();
  std::cout << str;
  if (validate) {
    std::cout.width(15);
    std::cout << "Test loss";
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
void Trainer::show_train_info(real_t tr_loss, real_t tr_metric,
                              real_t te_loss, real_t te_metric,
                              real_t time_cost, bool validate,
                              index_t epoch) {
  std::cout.width(6);
  std::cout << epoch;
  std::cout.width(15);
  std::cout << std::fixed << std::setprecision(5) << tr_loss;
  std::cout.width(20);
  std::cout << std::fixed << std::setprecision(5) << tr_metric;
  if (validate) {
    std::cout.width(15);
    std::cout << std::fixed << std::setprecision(5) << te_loss;
    std::cout.width(20);
    std::cout << std::fixed << std::setprecision(5) << te_metric;
  }
  std::cout.width(19);
  std::cout << std::fixed << std::setprecision(2) << time_cost;
  std::cout << std::endl;
}

// The basic
void Trainer::Train() {
  // show head info
  bool validate = test_reader_ == nullptr ? false : true;
  show_head_info(validate);
  //for in n epoch
  for (int n = 0; n < epoch_; ++n) {
    clock_t start, end;
    start = clock();
    //----------------------------------------------------
    // Calc grad and update model
    //----------------------------------------------------
    CalcGrad_Update();
    //----------------------------------------------------
    // Calc Train loss
    //----------------------------------------------------
    real_t tr_loss = 0.0;
    real_t tr_metric = 0.0;
    CalcLoss_Metric(train_reader_, &tr_loss, &tr_metric);
    //----------------------------------------------------
    // Calc Test loss
    //----------------------------------------------------
    real_t te_loss = 0.0;
    real_t te_metric = 0.0;
    if (validate) {
      CalcLoss_Metric(test_reader_, &te_loss, &te_metric);
    }
    end = clock();
    real_t time_cost = (real_t)(end-start) / CLOCKS_PER_SEC;
    // show train info
    show_train_info(tr_loss, tr_metric,
                    te_loss, te_metric,
                    time_cost, validate, n);
  }
}

// Training using cross-validation
void Trainer::CVTrain() {
  std::vector<real_t> train_loss(reader_list_.size());
  std::vector<real_t> test_loss(reader_list_.size());
  // Use the i-th Reader as validation Reader
  for (int i = 0; i < reader_list_.size(); ++i) {
    printf("  Cross-validation: %d/%lu: \n", i+1, reader_list_.size());
    // for n epoch
    for (int n = 0; n < epoch_; ++n) {
      // Use the other Reader to train
      for (int j = 0; j < reader_list_.size(); ++j) {
        if (i == j) { continue; }
        train_reader_ = reader_list_[j];
        train_reader_->Reset();
        DMatrix* matrix = NULL;
        while (train_reader_->Samples(matrix)) {
          loss_->CalcGrad(matrix, *model_);
        }
      }
      // Calc train loss
      index_t count_sample = 0;
      real_t loss_val = 0.0;
      std::vector<real_t> pred;
      int tmp = 0;
      for (int j = 0; j < reader_list_.size(); ++j) {
        if (i == j) { continue; }
        train_reader_ = reader_list_[j];
        train_reader_->Reset();
        DMatrix* matrix = NULL;
        while (1) {
          tmp = train_reader_->Samples(matrix);
          if (tmp == 0) { break; }
          if (tmp != pred.size()) {
            pred.resize(tmp);
          }
          count_sample += tmp;
          loss_->Predict(matrix, *model_, pred);
          loss_val += loss_->Evalute(pred, matrix->Y);
        }
      }
      loss_val /= count_sample;
      if (n == epoch_-1) {
        train_loss[i] = loss_val;
      }
      printf("    Epoch %d  |  Train loss: %f  |", n, loss_val);
      // Calc Test loss
      count_sample = 0;
      loss_val = 0.0;
      tmp = 0;
      test_reader_ = reader_list_[i];
      test_reader_->Reset();
      DMatrix* matrix = NULL;
      while (1) {
        tmp = test_reader_->Samples(matrix);
        if (tmp == 0) { break; }
        if (tmp != pred.size()) {
          pred.resize(tmp);
        }
        count_sample += tmp;
        loss_->Predict(matrix, *model_, pred);
        loss_val += loss_->Evalute(pred, matrix->Y);
      }
      loss_val /= count_sample;
      if (n == epoch_-1) {
        test_loss[i] = loss_val;
      }
      printf("  Test loss: %f  |", loss_val);
      //TIME_END();
      printf("  ");
      //SHOW_TIME();
      printf("  |\n");
    }
  }
  // average train loss
  real_t aver = 0;
  for (int i = 0; i < train_loss.size(); ++i) {
    aver += train_loss[i];
  }
  aver /= train_loss.size();
  printf("  Average train loss: %f \n", aver);
  // average test loss
  aver = 0;
  for (int i = 0; i < test_loss.size(); ++i) {
    aver += test_loss[i];
  }
  aver /= test_loss.size();
  printf("  Average test loss: %f \n", aver);
}

} // namespace xLearn
