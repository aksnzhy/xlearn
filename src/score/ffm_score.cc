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

#include "src/score/ffm_score.h"

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
  const real_t* data = (&((*w)[0])) + num_feature_;
  __MX _sum = _MMX_SETZERO_PS();
  // latent factor
  for (index_t i = 0; i < col_len; ++i) {
    real_t val_i = row->X[i];
    index_t idx_i = row->idx[i];
    index_t field_i = row->field[i];
    index_t mat_mul_pos_i = matrix_size * idx_i;
    index_t field_i_mul_fac = field_i * num_factor_;
    for (index_t j = i+1; j < col_len; ++j) {
      real_t val_j = row->X[j];
      index_t idx_j = row->idx[j];
      index_t field_j = row->field[j];
      const real_t* w_j = data + mat_mul_pos_i + num_factor_ * field_j;
      const real_t* w_i = data + matrix_size * idx_j + field_i_mul_fac;
      __MX _v = _MMX_SET1_PS(val_j*val_i);
      for (index_t k = 0; k < num_factor_; k += _MMX_INCREMENT) {
        __MX _kj = _MMX_LOAD_PS(w_j + k);
        __MX _ki = _MMX_LOAD_PS(w_i + k);
        _sum = _MMX_ADD_PS(_sum,
               _MMX_MUL_PS(_MMX_MUL_PS(_kj, _ki), _v));
      }
    }
  }
  real_t sum = 0;
#ifdef __AVX__
  sum += sum8(_sum);
#else // SSE
  _sum = _mm_hadd_ps(_sum, _sum);
  _sum = _mm_hadd_ps(_sum, _sum);
  _mm_store_ss(&sum, _sum);
#endif
  score += sum;
  return score;
}

// Calculate gradient and update current model parameters.
// Using sse/avx to speed up
void FFMScore::CalcGrad(const SparseRow* row,
                        std::vector<real_t>& param,
                        real_t pg, /* partial gradient */
                        Updater* updater) {
  static size_t matrix_size = num_factor_ * num_field_;
  size_t col_len = row->column_len;
  // for linear term
  for (size_t i = 0; i < col_len; ++i) {
    real_t gradient = pg * row->X[i];
    updater->Update(row->idx[i], gradient, param);
  }
  // for latent factor
  const real_t* data = &(param[0]) + num_feature_;
  for (size_t i = 0; i < col_len; ++i) {
    //------------------------- feat_i --------------------//
    real_t x_i = row->X[i];
    index_t idx_i = row->idx[i];
    index_t field_i = row->field[i];
    //------------------------- tmp ----------------------//
    index_t mat_mul_pos_i = matrix_size * idx_i;
    index_t field_i_mul_fac = field_i * num_factor_;
    real_t tmp = x_i * pg;
    for (size_t j = i+1; j < col_len; ++j) {
      //--------------------- feat_j ------------------------//
      real_t x_j = row->X[j];
      index_t idx_j = row->idx[j];
      index_t field_j = row->field[j];
      const real_t* w_j = data + mat_mul_pos_i + num_factor_ * field_j;
      const real_t* w_i = data + matrix_size * idx_j + field_i_mul_fac;
      __MX _v = _MMX_SET1_PS(x_j*tmp);
      for (size_t k = 0; k < num_factor_; k += _MMX_INCREMENT) {
        //----------- gradient_i and gradient_j -------------//
        /*
        _MMX_STORE_PS(p_i+k,
                      _MMX_MUL_PS(val_tmp,
                      _MMX_LOAD_PS(array_i + k)));
        _MMX_STORE_PS(p_j+k,
                      _MMX_MUL_PS(val_tmp,
                      _MMX_LOAD_PS(array_j + k)));*/
        real_t* w_i_k = const_cast<real_t*>(w_i + k);
        real_t* w_j_k = const_cast<real_t*>(w_j + k);
        __MX _kj = _MMX_LOAD_PS(w_j_k);
        __MX _ki = _MMX_LOAD_PS(w_i_k);
        __MX _gj = _MMX_MUL_PS(_v, _ki);
        __MX _gi = _MMX_MUL_PS(_v, _kj);
        updater->BatchUpdate(_kj, _gj, w_j_k);
        updater->BatchUpdate(_ki, _gi, w_i_k);
        //_MMX_STORE_PS(w_j_k, _MMX_SUB_PS(_kj, _MMX_MUL_PS(_gj, _lr)));
        //_MMX_STORE_PS(w_i_k, _MMX_SUB_PS(_ki, _MMX_MUL_PS(_gi, _lr)));
      }
    }
  }
}

} // namespace xLearn
