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

// Given predictions and labels, return squared loss value
real_t SquaredLoss::Evalute(const std::vector<real_t>& pred,
                            const std::vector<real_t>& label) {
  CHECK_EQ(pred.empty(), false);
  real_t val = 0.0;
  for (size_t i = 0; i < pred.size(); ++i) {
    real_t error = label[i] - pred[i];
    val += (error*error);
  }
  return val * 0.5;
}

// Calculate gradient in one thread
void squared_thread(const DMatrix* matrix,
                    Model* model,
                    Score* score_func,
                    bool is_norm,
                    index_t start,
                    index_t end) {
  CHECK_GT(end, start);
  for (size_t i = start; i < end; ++i) {
    SparseRow* row = matrix->row[i];
    real_t norm = is_norm ? matrix->norm[i] : 1.0;
    real_t score = score_func->CalcScore(row, *model, norm);
    // partial gradient: -error
    real_t pg = score - matrix->Y[i];
    // real gradient and update
    score_func->CalcGrad(row, *model, pg, norm);
  }
}

// Calculate gradient in multi-thread
void SquaredLoss::CalcGrad(const DMatrix* matrix,
                           Model& model) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_length, 0);
  size_t row_len = matrix->row_length;
  for (int i = 0; i < threadNumber_; ++i) {
    size_t start = getStart(row_len, threadNumber_, i);
    size_t end = getEnd(row_len, threadNumber_, i);
    pool_->enqueue(std::bind(squared_thread,
                             matrix,
                             &model,
                             score_func_,
                             norm_,
                             start,
                             end));
  }
  pool_->Sync();
}

} // namespace xLearn
