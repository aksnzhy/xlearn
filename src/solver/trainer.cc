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

// Standard training
void Trainer::Train() {
  //for in n epoch
  for (int n = 0; n < epoch_; ++n) {
    TIME_START();
    // Return to the head of the data
    train_reader_->Reset();
    if (test_reader_ != NULL) {
      test_reader_->Reset();
    }
    DMatrix* matrix = NULL;
    // Calc grad and update model
    clock_t start_2, end_2;
    start_2 = clock();
    while (train_reader_->Samples(matrix)) {
      loss_->CalcGrad(matrix, model_, updater_);
    }
    end_2 = clock();
    printf("Time: %.2f sec", (float)(end_2-start_2) / CLOCKS_PER_SEC);
    // Calc Train loss
    train_reader_->Reset();
    index_t count_sample = 0;
    real_t loss_val = 0.0;
    std::vector<real_t> pred;
    int tmp = 0;
    while (1) {
      tmp = train_reader_->Samples(matrix);
      if (tmp == 0) { break; }
      if (tmp != pred.size()) {
        pred.resize(tmp);
      }
      count_sample += tmp;
      loss_->Predict(matrix, model_, pred);
      loss_val += loss_->Evalute(pred, matrix->Y);
    }
    loss_val /= count_sample;
    printf("  Epoch %d  |  Train loss: %f  |", n, loss_val);
    // Calc Test loss
    if (test_reader_ != NULL) {
      count_sample = 0;
      loss_val = 0;
      tmp = 0;
      while (1) {
        tmp = test_reader_->Samples(matrix);
        if (tmp == 0) { break; }
        if (tmp != pred.size()) {
          pred.resize(tmp);
        }
        count_sample += tmp;
        loss_->Predict(matrix, model_, pred);
        loss_val += loss_->Evalute(pred, matrix->Y);
      }
      loss_val /= count_sample;
      printf("  Test loss: %f  |", loss_val);
    }
    TIME_END();
    printf("  ");
    SHOW_TIME();
    printf("  |\n");
  }
}

// Training using cross-validation
void Trainer::CVTrain() {
  std::vector<real_t> train_loss(reader_list_.size());
  std::vector<real_t> test_loss(reader_list_.size());
  // Use ith Reader as validation Reader
  for (int i = 0; i < reader_list_.size(); ++i) {
    printf("  Cross-validation: %d/%lu: \n", i+1, reader_list_.size());
    // for n epoch
    for (int n = 0; n < epoch_; ++n) {
      TIME_START();
      // Use the other Reader to train
      for (int j = 0; j < reader_list_.size(); ++j) {
        if (i == j) { continue; }
        train_reader_ = reader_list_[j];
        train_reader_->Reset();
        DMatrix* matrix = NULL;
        while (train_reader_->Samples(matrix)) {
          loss_->CalcGrad(matrix, model_, updater_);
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
          loss_->Predict(matrix, model_, pred);
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
        loss_->Predict(matrix, model_, pred);
        loss_val += loss_->Evalute(pred, matrix->Y);
      }
      loss_val /= count_sample;
      if (n == epoch_-1) {
        test_loss[i] = loss_val;
      }
      printf("  Test loss: %f  |", loss_val);
      TIME_END();
      printf("  ");
      SHOW_TIME();
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
