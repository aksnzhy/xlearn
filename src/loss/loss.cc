//------------------------------------------------------------------------------
// Copyright (c) 2018 by contributors. All Rights Reserved.
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
This file is the implementation of the basic Loss class.
*/

#include "src/loss/loss.h"
#include "src/loss/squared_loss.h"
#include "src/loss/cross_entropy_loss.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_loss_registry, Loss);
REGISTER_LOSS("squared", SquaredLoss);
REGISTER_LOSS("cross-entropy", CrossEntropyLoss);

// Predict in one thread
void pred_thread(const DMatrix* matrix,
                 Model* model,
                 std::vector<real_t>* pred,
                 Score* score_func_,
                 bool is_norm,
                 size_t start_idx,
                 size_t end_idx) {
  CHECK_GE(end_idx, start_idx);
  for (size_t i = start_idx; i < end_idx; ++i) {
    SparseRow* row = matrix->row[i];
    real_t norm = is_norm ? matrix->norm[i] : 1.0;
    (*pred)[i] = score_func_->CalcScore(row, *model, norm);
  }
}

// Predict in multi-thread
void Loss::Predict(const DMatrix* matrix,
                   Model& model,
                   std::vector<real_t>& pred) {
  CHECK_NOTNULL(matrix);
  CHECK_NE(pred.empty(), true);
  CHECK_EQ(pred.size(), matrix->row_length);
  index_t row_len = matrix->row_length;
  // Predict in multi-thread
  for (int i = 0; i < threadNumber_; ++i) {
    size_t start_idx = getStart(row_len, threadNumber_, i);
    size_t end_idx = getEnd(row_len, threadNumber_, i);
    pool_->enqueue(std::bind(pred_thread,
                             matrix,
                             &model,
                             &pred,
                             score_func_,
                             norm_,
                             start_idx,
                             end_idx));
  }
  // Wait all of the threads finish their job
  pool_->Sync(threadNumber_);
}

// Given data sample and current model, calculate gradient.
// Note that this method doesn't update local model, and the
// gradient will be pushed to the parameter server, which is 
// used for distributed computation.
void Loss::CalcGradDist(DMatrix* matrix,
                        Model& model,
                        std::vector<real_t>& grad) {
  for(;;) {
    // Get a mini-batch from current data matrix
    DMatrix mini_batch;
    mini_batch.ResetMatrix(batch_size_);
    index_t len = matrix->GetMiniBatch(batch_size_, mini_batch);
    if (len == 0) {
      break;
    }
    mini_batch.row_length = len;
    // Compress the sparse data matrix and sparse model
    // parameter to dense format
    std::vector<index_t> feature_list;
    mini_batch.Compress(feature_list);
    /*
    // Pull the model parameter from parameter server
    store->pull(feature_list, model);

    // Calculate gradient
    this->CalcGrad(matrix, model, grad);

    // Push gradient to the parameter server
    ps->push(grad, feature_list);
    */
  }
}

}  // namespace xLearn
