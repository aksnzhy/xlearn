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

#ifdef __AVX__

float sum8(__m256 x) {
  // hiQuad = ( x7, x6, x5, x4 )
  const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
  // loQuad = ( x3, x2, x1, x0 )
  const __m128 loQuad = _mm256_castps256_ps128(x);
  // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
  const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
  // loDual = ( -, -, x1 + x5, x0 + x4 )
  const __m128 loDual = sumQuad;
  // hiDual = ( -, -, x3 + x7, x2 + x6 )
  const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
  // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
  const __m128 sumDual = _mm_add_ps(loDual, hiDual);
  // lo = ( -, -, -, x0 + x2 + x4 + x6 )
  const __m128 lo = sumDual;
  // hi = ( -, -, -, x1 + x3 + x5 + x7 )
  const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
  const __m128 sum = _mm_add_ss(lo, hi);
  return _mm_cvtss_f32(sum);
}

#endif

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
    real_t val_i = row->X[i];
    index_t idx_i = row->idx[i];
    index_t field_i = row->field[i];
    index_t mat_mul_pos_i = matrix_size * idx_i;
    index_t field_i_mul_fac = field_i * num_factor_;
    const real_t* data = (&((*w)[0])) + num_feature_;
    for (index_t j = i+1; j < col_len; ++j) {
      real_t val_j = row->X[j];
      index_t idx_j = row->idx[j];
      index_t field_j = row->field[j];
      __MX _accu = _MMX_SET1_PS(0);
      for (index_t k = 0; k < num_factor_; k += _MMX_INCREMENT) {
        __MX _kj = _MMX_LOAD_PS(data + mat_mul_pos_i
                                     + field_j
                                     * num_factor_);
        __MX _ki = _MMX_LOAD_PS(data + matrix_size
                                     * idx_j
                                     + field_i_mul_fac);
        __MX _tmp = _MMX_MUL_PS(_ki, _kj);
        _accu = _MMX_ADD_PS(_accu, _tmp);
      }
      // accumulate _accu to sum
      real_t sum = 0.0;
#ifdef __AVX__
      sum += sum8(_accu);
#else // SSE
      _accu = _mm_hadd_ps(_accu, _accu);
      _accu = _mm_hadd_ps(_accu, _accu);
      _mm_store_ss(&sum, _accu);
#endif
      sum *= val_i * val_j;
      score += sum;
    }
  }
  return score;
}

} // namespace xLearn
