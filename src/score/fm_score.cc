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
This file is the implementation of FMScore class.
*/

#include "src/score/fm_score.h"

namespace xLearn {

// y = wTx + sum[(V_i*V_j)(x_i * x_j)]
real_t FMScore::CalcScore(const SparseRow* row,
                          const std::vector<real_t>* w) {
  real_t score = 0.0;
  index_t col_len = row->column_len;
  // linear term
  for (index_t i = 0; i < col_len; ++i) {
    index_t pos = row->idx[i];
    score += (*w)[pos] * row->X[i];
  }
  // latent factor
  real_t tmp = 0.0;
  for (index_t k = 0; k < num_factor_; ++k) {
    real_t square_sum = 0.0, sum_square = 0.0;
    index_t tmp_idx = num_feature_ + k;
    // Skip the first element because it is the bias
    for (index_t i = 0; i < col_len; ++i) {
      real_t x = row->X[i];
      index_t pos = row->idx[i] * num_factor_ + tmp_idx;
      real_t v = (*w)[pos];
      square_sum += (x*v);
      sum_square += (x*x*v*v);
    }
    square_sum *= square_sum;
    tmp += (square_sum - sum_square);
  }
  score += 0.5 * tmp;
  return score;
}

// Calculate gradient and update current model parameters.
void FMScore::CalcGrad(const SparseRow* row,
                       std::vector<real_t>& param,
                       real_t pg, /* partial gradient */
                       Updater* updater) {
  size_t col_len = row->column_len;
  // for linear term
  for (size_t i = 0; i < col_len; ++i) {
    real_t gradient = pg * row->X[i];
    updater->Update(row->idx[i], gradient, param);
  }
  // for latent factor
  for (size_t k = 0; k < num_factor_; ++k) {
    real_t v_mul_x = 0.0;
    index_t tmp_idx = num_feature_ + k;
    for (index_t i = 0; i < col_len; ++i) {
      index_t pos = row->idx[i] * num_factor_ + tmp_idx;
      real_t v = param[pos];
      real_t x = row->X[i];
      v_mul_x += (x*v);
    }
    for (index_t i = 0; i < col_len; ++i) {
      index_t pos = row->idx[i] * num_factor_ + tmp_idx;
      real_t v = param[pos];
      real_t x = row->X[i];
      real_t gradient = (x*v_mul_x - v*x*x) * pg;
      updater->Update(pos, gradient, param);
    }
  }
}

} // namespace xLearn
