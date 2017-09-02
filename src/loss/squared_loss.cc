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
This file is the implementation of SquaredLoss class.
*/

#include "src/loss/squared_loss.h"

namespace xLearn {

// Given predictions and labels, return squared loss value.
real_t SquaredLoss::Evalute(const std::vector<real_t>& pred,
                            const std::vector<real_t>& label) {
  CHECK_EQ(pred.empty(), false);
  CHECK_EQ(pred.size(), label.size());
  real_t val = 0.0;
  for (size_t i = 0; i < pred.size(); ++i) {
    real_t tmp = pred[i] - label[i];
    val += 0.5*(tmp*tmp);
  }
  val /= pred.size();
  return val;
}

// Given data sample and current model, calculate gradient
// and update model.
void SquaredLoss::CalcGrad(const DMatrix* matrix,
                           Model* model,
                           Updater* updater) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_len, 0);
  CHECK_NOTNULL(updater);
  std::vector<real_t>* w = model->GetParameter();
  size_t row_len = matrix->row_len;
  // Calculate gradient
  for (size_t i = 0; i < row_len; ++i) {
    SparseRow* row = matrix->row[i];
    real_t score = score_func_->CalcScore(row, w);
    // partial gradient
    real_t pg = score - matrix->Y[i];
    // real gradient and update
    score_func_->CalcGrad(row, *w, pg, updater);
  }
}

} // namespace xLearn
