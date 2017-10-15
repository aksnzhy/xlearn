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
#include "src/base/math.h"

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
                           Model& model) {
  real_t score = 0.0;
  /*********************************************************
   *  linear term                                          *
   *********************************************************/
  static real_t *w = model.GetParameter_w();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t pos = iter->feat_id;
    score += w[pos] * iter->feat_val;
  }
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  static index_t align0 = model.GetNumK();
  static index_t align1 = model.GetNumField() * model.GetNumK();
  __MX _sum = _MMX_SETZERO_PS();
  w = model.GetParameter_w() + model.GetNumFeature();
  for (SparseRow::const_iterator iter_i = row->begin();
       iter_i != row->end(); ++iter_i) {
    real_t v1 = iter_i->feat_val;
    index_t j1 = iter_i->feat_id;
    index_t f1 = iter_i->field_id;
    for (SparseRow::const_iterator iter_j = iter_i+1;
         iter_j != row->end(); ++iter_j) {
      real_t v2 = iter_j->feat_val;
      index_t j2 = iter_j->feat_id;
      index_t f2 = iter_j->field_id;
      real_t* w1_base = w + j1*align1 + f2*align0;
      real_t* w2_base = w + j2*align1 + f1*align0;
      __MX _v = _MMX_SET1_PS(v1*v2);
      for (index_t k = 0; k < align0; k += _MMX_INCREMENT) {
        __MX _w1 = _MMX_LOAD_PS(w1_base + k);
        __MX _w2 = _MMX_LOAD_PS(w2_base + k);
        _sum = _MMX_ADD_PS(_sum,
               _MMX_MUL_PS(_MMX_MUL_PS(_w1, _w2), _v));
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

// Calculate gradient and update current model
// parameters. Using sse/avx to speed up
void FFMScore::CalcGrad(const SparseRow* row,
                        Model& model,
                        real_t pg) {
  /*********************************************************
   *  linear term                                          *
   *********************************************************/
  static real_t *w = model.GetParameter_w();
  static real_t* cache = model.GetParameter_cache();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    real_t gradient = pg * iter->feat_val;
    index_t idx = iter->feat_id;
    gradient += regu_lambda_ * w[idx];
    cache[idx] += (gradient * gradient);
    w[idx] -= (learning_rate_ * gradient *
               InvSqrt((cache)[idx]));
  }
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  static index_t align0 = model.GetNumK();
  static index_t align1 = model.GetNumField() * model.GetNumK();
  static __MX _pg = _MMX_SET1_PS(pg);
  static __MX _lr = _MMX_SET1_PS(learning_rate_);
  static __MX _lamb = _MMX_SET1_PS(regu_lambda_);
  w = model.GetParameter_w() + model.GetNumFeature();
  cache = model.GetParameter_cache() + model.GetNumFeature();
  for (SparseRow::const_iterator iter_i = row->begin();
       iter_i != row->end(); ++iter_i) {
    real_t v1 = iter_i->feat_val;
    index_t j1 = iter_i->feat_id;
    index_t f1 = iter_i->field_id;
    for (SparseRow::const_iterator iter_j = iter_i+1;
         iter_j != row->end(); ++iter_j) {
      real_t v2 = iter_j->feat_val;
      index_t j2 = iter_j->feat_id;
      index_t f2 = iter_j->field_id;
      index_t bias_1 = j1*align1 + f2*align0;
      index_t bias_2 = j2*align1 + f1*align0;
      real_t* w1_base = w + bias_1;
      real_t* w2_base = w + bias_2;
      real_t* c1_base = cache + bias_1;
      real_t* c2_base = cache + bias_2;
      __MX _v = _MMX_SET1_PS(v1*v2);
      __MX _v_pg = _MMX_MUL_PS(_v, _pg);
      for (size_t k = 0; k < align0; k += _MMX_INCREMENT) {
        real_t* w1 = w1_base + k;
        real_t* w2 = w2_base + k;
        real_t* c1 = c1_base + k;
        real_t* c2 = c2_base + k;
        __MX _w1 = _MMX_LOAD_PS(w1);
        __MX _w2 = _MMX_LOAD_PS(w2);
        __MX _c1 = _MMX_LOAD_PS(c1);
        __MX _c2 = _MMX_LOAD_PS(c2);
        __MX _g1 = _MMX_ADD_PS(
                   _MMX_MUL_PS(_lamb, _w1),
                   _MMX_MUL_PS(_v_pg, _w2));
        __MX _g2 = _MMX_ADD_PS(
                   _MMX_MUL_PS(_lamb, _w2),
                   _MMX_MUL_PS(_v_pg, _w1));
        _c1 = _MMX_ADD_PS(_c1, _MMX_MUL_PS(_g1, _g1));
        _c2 = _MMX_ADD_PS(_c2, _MMX_MUL_PS(_g2, _g2));
        _w1 = _MMX_SUB_PS(_w1, _MMX_MUL_PS(_lr,
              _MMX_MUL_PS(_MMX_RSQRT_PS(_c1), _g1)));
        _w2 = _MMX_SUB_PS(_w2, _MMX_MUL_PS(_lr,
              _MMX_MUL_PS(_MMX_RSQRT_PS(_c2), _g2)));
        _MMX_STORE_PS(w1, _w1);
        _MMX_STORE_PS(w2, _w2);
        _MMX_STORE_PS(c1, _c1);
        _MMX_STORE_PS(c2, _c2);
      }
    }
  }
}

} // namespace xLearn
