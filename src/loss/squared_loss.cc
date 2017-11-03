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

// Calculate loss in one thread.
static void sq_evalute_thread(const std::vector<real_t>* pred,
                              const std::vector<real_t>* label,
                              real_t* tmp_sum,
                              size_t start_idx,
                              size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  *tmp_sum = 0;
  for (size_t i = start_idx; i < end_idx; ++i) {
    real_t error = (*label)[i] - (*pred)[i];
    (*tmp_sum) += (error*error);
  }
  *tmp_sum *= 0.5;
}

//------------------------------------------------------------------------------
// Calculate loss in multi-thread:
//
//                         master_thread
//                      /       |         \
//                     /        |          \
//                thread_1    thread_2    thread_3
//                   |           |           |
//                    \          |           /
//                     \         |          /
//                       \       |        /
//                         master_thread
//------------------------------------------------------------------------------
real_t SquaredLoss::Evalute(const std::vector<real_t>& pred,
                            const std::vector<real_t>& label) {
  CHECK_NE(pred.empty(), true);
  CHECK_NE(label.empty(), true);
  real_t val = 0.0;
  // multi-thread training
  std::vector<real_t> sum(threadNumber_, 0);
  for (int i = 0; i < threadNumber_; ++i) {
    size_t start_idx = getStart(pred.size(), threadNumber_, i);
    size_t end_idx = getEnd(pred.size(), threadNumber_, i);
    pool_->enqueue(std::bind(sq_evalute_thread,
                             &pred,
                             &label,
                             &(sum[i]),
                             start_idx,
                             end_idx));
  }
  // Wait all of the threads finish their job
  pool_->Sync();
  for (size_t i = 0; i < sum.size(); ++i) {
    val += sum[i];
  }
  return val;
}

// Calculate gradient in one thread
void sq_gradient_thread(const DMatrix* matrix,
                        Model* model,
                        Score* score_func,
                        bool is_norm,
                        index_t start,
                        index_t end) {
  CHECK_GE(end, start);
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

//------------------------------------------------------------------------------
// Calculate gradient in multi-thread
//
//                         master_thread
//                      /       |         \
//                     /        |          \
//                thread_1    thread_2    thread_3
//                   |           |           |
//                    \          |           /
//                     \         |          /
//                       \       |        /
//                         master_thread
//------------------------------------------------------------------------------
void SquaredLoss::CalcGrad(const DMatrix* matrix,
                           Model& model) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_length, 0);
  size_t row_len = matrix->row_length;
  for (int i = 0; i < threadNumber_; ++i) {
    size_t start = getStart(row_len, threadNumber_, i);
    size_t end = getEnd(row_len, threadNumber_, i);
    pool_->enqueue(std::bind(sq_gradient_thread,
                             matrix,
                             &model,
                             score_func_,
                             norm_,
                             start,
                             end));
  }
  // Wait all of the threads finish their job
  pool_->Sync();
}

} // namespace xLearn
