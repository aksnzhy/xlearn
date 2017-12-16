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

#include "src/distributed/dist_cross_entropy_loss.h"

#include<thread>
#include<atomic>
#include<unordered_map>

namespace xLearn {

// Calculate loss in one thread.
static void ce_evalute_thread(const std::vector<real_t>* pred,
                              const std::vector<real_t>* label,
                              real_t* tmp_sum,
                              size_t start_idx,
                              size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  *tmp_sum = 0;
  for (size_t i = start_idx; i < end_idx; ++i) {
    real_t y = (*label)[i] > 0 ? 1.0 : -1.0;
    (*tmp_sum) += log1p(exp(-y*(*pred)[i]));
  }
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
void DistCrossEntropyLoss::Evalute(const std::vector<real_t>& pred,
                               const std::vector<real_t>& label) {
  CHECK_NE(pred.empty(), true);
  CHECK_NE(label.empty(), true);
  total_example_ += pred.size();
  // multi-thread training
  std::vector<real_t> sum(threadNumber_, 0);
  for (int i = 0; i < threadNumber_; ++i) {
    size_t start_idx = getStart(pred.size(), threadNumber_, i);
    size_t end_idx = getEnd(pred.size(), threadNumber_, i);
    pool_->enqueue(std::bind(ce_evalute_thread,
                             &pred,
                             &label,
                             &(sum[i]),
                             start_idx,
                             end_idx));
  }
  // Wait all of the threads finish their job
  pool_->Sync(threadNumber_);
  // Accumulate loss
  for (size_t i = 0; i < sum.size(); ++i) {
    loss_sum_ += sum[i];
  }
}


// Calculate gradient in one thread.
static void ce_gradient_thread(const DMatrix* matrix,
                               std::unordered_map<index_t, real_t>& w,
                               DistScore* dist_score_func,
                               bool is_norm,
                               real_t* sum,
                               std::unordered_map<index_t, real_t>& g,
                               size_t start_idx,
                               size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  dist_score_func->DistCalcGrad(matrix, w, sum, g, start_idx, end_idx);
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
void DistCrossEntropyLoss::CalcGrad(const DMatrix* matrix,
                                Model& model) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_length, 0);
  size_t row_len = matrix->row_length;
  total_example_ += row_len;
  auto feature_ids = std::vector<ps::Key>();
  auto gradient_pull = std::make_shared<std::vector<float>>();
  std::unordered_map<index_t, real_t> weight_map;
  std::unordered_map<index_t, real_t> gradient_push_map;
  std::vector<real_t> gradient_push;
  for (int i = 0; i < row_len; ++i) {
    SparseRow* row = matrix->row[i];
    for (SparseRow::const_iterator iter = row->begin();
         iter != row->end(); ++iter) {
      index_t idx = iter->feat_id;
      feature_ids.push_back(idx);
      weight_map[idx] = 0.0;
    }
  }
  std::sort(feature_ids.begin(), feature_ids.end());
  feature_ids.erase(unique(feature_ids.begin(), feature_ids.end()),
                      feature_ids.end());

  gradient_pull->resize(feature_ids.size());
  if (model.GetScoreFunction().compare("dist_linear") == 0) {
    kv_w_->Pull(feature_ids, &(*gradient_pull));
  }
  for (int i = 0; i < gradient_pull->size(); ++i) {
    index_t idx = feature_ids[i];
    real_t weight = (*gradient_pull)[i];
    weight_map[idx] = weight;
  }
  // multi-thread training
  int count = lock_free_ ? threadNumber_ : 1;
  std::vector<real_t> sum(count, 0);
  for (int i = 0; i < count; ++i) {
    index_t start_idx = getStart(row_len, count, i);
    index_t end_idx = getEnd(row_len, count, i);
    pool_->enqueue(std::bind(ce_gradient_thread,
                             matrix,
                             std::ref(weight_map),
                             dist_score_func_,
                             norm_,
                             &(sum[i]),
                             std::ref(gradient_push_map),
                             start_idx,
                             end_idx));
  }
  // Wait all of the threads finish their job
  pool_->Sync(count);
  for (int i = 0; i < feature_ids.size(); ++i) {
    index_t idx = feature_ids[i];
    real_t g = gradient_push_map[idx];
    gradient_pull->push_back(g);
  }
  if (model.GetScoreFunction().compare("dist_linear") == 0) {
    kv_w_->Push(feature_ids, gradient_push);
  }
  // Accumulate loss
  for (int i = 0; i < sum.size(); ++i) {
    loss_sum_ += sum[i];
  }
}

} // namespace xLearn
