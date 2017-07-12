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
This file is the implementation of FFMScore class.
*/

#include "src/loss/ffm_score.h"

namespace xLearn {

// y = wTx + sum[(V_i_fj*V_j_fi)(x_i * x_j)]
// Using sse/avx to speed up
real_t FFMScore::CalcScore(const SparseRow* row,
                           const std::vector<real_t>* w) {
  static index_t matrix_size = num_field_ * num_factor_;
  real_t score = 0.0;
  index_t col_len = row->column_len;
  // linear term： wTx
  for (index_t i = 0; i < col_len; ++i) {
    index_t pos = row->idx[i];
    score += (*w)[pos] * row->X[i];
  }
  // latent factor
  for (index_t i = 0; i < col_len; ++i) {
    index_t pos_i = row->idx[i];
    real_t val_i = row->X[i];
    index_t field_i = row->field[i];
    index_t mat_mul_pos_i = matrix_size * pos_i;
    index_t field_i_mul_fac = field_i * num_factor_;
    const real_t* data = w->data() + num_feature_;
    for (index_t j = i+1; j < col_len; ++j) {
      index_t pos_j = row->idx[j];
      real_t val_j = row->X[j];
      index_t field_j = row->field[j];
      const real_t* K_i = data + mat_mul_pos_i
                          + field_j*num_factor_;
      const real_t* K_j = data + matrix_size*pos_j
                          + field_i_mul_fac;
      real_t tmp = 0.0;
      for (index_t k = 0; k < num_factor_; ++k) {
        tmp += (*(K_i+k)) * (*(K_j+k));
      }
      tmp = tmp * val_i * val_j;
      score += tmp;
    }
  }
  return score;
}

} // namespace xLearn
