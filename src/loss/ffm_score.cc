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
  // linear term
  for (index_t i = 0; i < col_len; ++i) {
    index_t pos = row->idx[i];
    score += (*w)[pos] * row->X[i];
  }
  // latent factor
  const float* array_ = &((*w)[0]);
  __MX accumulate_ = _MMX_SETZERO_PS();
  for (index_t i = 1; i < col_len; ++i) {
    index_t idx_i_mul_mxs_add_mf = row->idx[i] * matrix_size + max_feature_;
    index_t field_i_mul_fac_mf = row->field[i] * num_factor_ + max_feature_;
    real_t x_i = row->X[i];
    for (index_t j = i+1; j < col_len; ++j) {
      index_t pos_i = idx_i_mul_mxs_add_mf + row->field[j] * num_factor_;
      index_t pos_j = row->idx[j] * matrix_size + field_i_mul_fac_mf;
      real_t x_i_mul_x_j = row->X[j] * x_i;
      __MX x_i_x_j = _MMX_SET1_PS(x_i_mul_x_j);
      for (index_t k = 0; k < num_factor_; k += _MMX_INCREMENT) {
        __MX wi = _MMX_LOAD_PS(array_ + pos_i + k);
        __MX wj = _MMX_LOAD_PS(array_ + pos_j + k);
        accumulate_ = _MMX_ADD_PS(accumulate_,
                      _MMX_MUL_PS(wi,
                      _MMX_MUL_PS(wj, x_i_x_j)));
      }
    }
  }
#ifdef __AVX__
  accumulate_ = _MMX_HADD_PS(accumulate_, accumulate_);
  real_t tmp[8];
  _MMX_STORE_SS(tmp, accumulate_);
  score = tmp[0];
#else
  _MMX_STORE_SS(&score, accumulate_);
#endif
  return score;
}

} // namespace xLearn
