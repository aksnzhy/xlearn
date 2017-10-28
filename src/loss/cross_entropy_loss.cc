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
This file is the implementation of CrossEntropyLoss class.
*/

#include "src/loss/cross_entropy_loss.h"

#include <thread>
#include<atomic>

namespace xLearn {

// Given predictions (data samples) and labels, return
// cross-entropy loss value
real_t CrossEntropyLoss::Evalute(const std::vector<real_t>& pred,
                                 const std::vector<real_t>& label) {
  CHECK_NE(pred.empty(), true);
  CHECK_NE(label.empty(), true);
  real_t val = 0.0;
  for (size_t i = 0; i < pred.size(); ++i) {
    real_t y = label[i] > 0 ? 1.0 : -1.0;
    val += log1p(exp(-y*pred[i]));
  }
  return val;
}

// Calculate gradient in one thread
void cross_entropy_thread(const DMatrix* matrix,
                        Model* model,
                        Score* score_func,
                        bool is_norm,
                        index_t start,
                        index_t end) {
  CHECK_GT(end, start);
  for (index_t i = start; i < end; ++i) {
    SparseRow* row = matrix->row[i];
    real_t norm = is_norm ? matrix->norm[i] : 1.0;
    real_t score = score_func->CalcScore(row, *model, norm);
    // partial gradient
    real_t y = matrix->Y[i] > 0 ? 1.0 : -1.0;
    real_t pg = -y/(1.0+(1.0/exp(-y*score)));
    // real gradient and update
    score_func->CalcGrad(row, *model, pg, norm);
  }
}

// Calculate gradient in multi-thread
void CrossEntropyLoss::CalcGrad(const DMatrix* matrix,
                                Model& model) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_length, 0);
  index_t row_len = matrix->row_length;
  // multi-thread training
  for (int i = 0; i < threadNumber_; ++i) {
    index_t start = getStart(row_len, threadNumber_, i);
    index_t end = getEnd(row_len, threadNumber_, i);
    pool_->enqueue(std::bind(cross_entropy_thread,
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
