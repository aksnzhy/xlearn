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
    // Return to the head of the data
    train_reader_->Reset();
    if (test_reader_ != NULL) {
      test_reader_->Reset();
    }
    DMatrix* matrix = NULL;
    // Calc grad and update model
    while (train_reader_->Samples(matrix)) {
      loss_->CalcGrad(matrix, model_, updater_);
    }
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
    printf("  Epoch %d: |  Train loss: %f  |", n, loss_val);
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
      printf("   Test loss: %f  |", loss_val);
    }
    printf("\n");
  }
}

// Training using cross-validation
void Trainer::CVTrain() {

}

// Save model to disk file
size_t Trainer::SaveModel() {
  size_t model_size = 0;

  return model_size;
}

} // namespace xLearn
