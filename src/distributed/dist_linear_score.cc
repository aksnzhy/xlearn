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
This file is the implementation of LinearScore class.
*/

#include "src/distributed/dist_linear_score.h"
#include "src/base/math.h"

namespace xLearn {

// y = wTx (incluing bias term)
real_t DistLinearScore::CalcScore(const SparseRow* row,
                                  std::unordered_map<index_t, real_t>* w,
                                  real_t norm) {
  real_t score = 0.0;
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t idx = iter->feat_id;
    score += (*w)[idx] * iter->feat_val;
  }
  return score;
}

// Calculate gradient and update current model
void DistLinearScore::DistCalcGrad(const DMatrix* matrix,
                           std::unordered_map<index_t, real_t>* w,
                           real_t* sum,
                           std::unordered_map<index_t, real_t>* g,
                           index_t start_idx,
                           index_t end_idx) {
  // Using sgd
  if (opt_type_.compare("sgd") == 0) {
    this->calc_grad_sgd(matrix, w, sum, g, start_idx, end_idx);
  }
  // Using adagrad
  else if (opt_type_.compare("adagrad") == 0) {
    this->calc_grad_adagrad(matrix, w, sum, g, start_idx, end_idx);
  }
  // Using ftrl
  else if (opt_type_.compare("ftrl") == 0) {
    this->calc_grad_ftrl(matrix, w, sum, g, start_idx, end_idx);
  }
}

// Calculate gradient and update current model using sgd
void DistLinearScore::calc_grad_sgd(const DMatrix* matrix,
                                std::unordered_map<index_t, real_t>* w,
                                real_t* sum,
                                std::unordered_map<index_t, real_t>* g,
                                real_t start_idx,
                                real_t end_idx) {
  // linear term
  for (index_t i = start_idx; i < end_idx; ++i) {
    SparseRow* row = matrix->row[i];
    real_t pred = CalcScore(row, w);
    real_t y = matrix->Y[i] > 0 ? 1.0 : -1.0;
    (*sum) += log1p(exp(-y*pred));
    real_t pg = -y / (1.0 + (1.0 / exp(-y * pred)));
    for (SparseRow::const_iterator iter = row->begin();
        iter != row->end(); ++iter) {
      index_t idx_g = iter->feat_id;
      real_t gradient = pg * iter->feat_val;
      gradient += regu_lambda_ * (*w)[idx_g]; // get gradient
      (*g)[idx_g] += gradient;
    }
  }
}

// Calculate gradient and update current model using adagrad
void DistLinearScore::calc_grad_adagrad(const DMatrix* matrix,
                                    std::unordered_map<index_t, real_t>* w,
                                    real_t* sum,
                                    std::unordered_map<index_t, real_t>* g,
                                    real_t start_idx,
                                    real_t end_idx) {
  // linear term
  for (index_t i = start_idx; i < end_idx; ++i) {
    SparseRow* row = matrix->row[i];
    real_t pred = CalcScore(row, w);
    real_t y = matrix->Y[i] > 0 ? 1.0 : -1.0;
    (*sum) += log1p(exp(-y*pred));
    real_t pg = -y / (1.0 + (1.0 / exp(-y * pred)));
    for (SparseRow::const_iterator iter = row->begin();
        iter != row->end(); ++iter) {
      index_t idx_g = iter->feat_id;
      real_t gradient = pg * iter->feat_val;
      gradient += regu_lambda_ * (*w)[idx_g]; // get gradient
      (*g)[idx_g] += gradient;
    }
  }
}

// Calculate gradient and update current model using ftrl
void DistLinearScore::calc_grad_ftrl(const DMatrix* matrix,
                                 std::unordered_map<index_t, real_t>* w,
                                 real_t* sum,
                                 std::unordered_map<index_t, real_t>* g,
                                 real_t start_idx,
                                 real_t end_idx) {
  for (int i = 0; i < start_idx; ++i) {
    SparseRow* row = matrix->row[i];
    real_t pred = CalcScore(row, w);
    real_t y = matrix->Y[i] > 0 ? 1.0 : -1.0;
    (*sum) += log1p(exp(-y * pred));
    real_t pg = -y / (1.0 + (1.0 / exp(-y * pred)));
    for (SparseRow::const_iterator iter = row->begin();
        iter != row->end(); ++iter) {
      index_t idx_g = iter->feat_id;
      real_t gradient = pg * iter->feat_val; // get gradient
      (*g)[idx_g] += gradient;
    }
  }
}

} // namespace xLearn
