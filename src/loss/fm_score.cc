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

#include "src/loss/fm_score.h"

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
    real_t square_sum = 0.0, sum_sqaure = 0.0;
    index_t tmp_idx = max_feature_ + k;
    for (index_t i = 0; i < col_len; ++i) {
      real_t x = row->X[i];
      index_t pos = row->idx[i] * num_factor_ + tmp_idx;
      real_t v = (*w)[pos];
      square_sum += (x*v);
      sum_sqaure += (x*x*v*v);
    }
    square_sum *= square_sum;
    tmp += (square_sum - sum_sqaure);
  }
  score += 0.5 * tmp;
  return score;
}

} // namespace xLearn
