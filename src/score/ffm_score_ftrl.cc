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

#include <pmmintrin.h>  // for SSE

#include "src/score/ffm_score_ftrl.h"
#include "src/base/math.h"

namespace xLearn {

// y = sum( (V_i_fj*V_j_fi)(x_i * x_j) )
// Using SSE to accelerate vector operation.
real_t FFMScoreFtrl::CalcScore(const SparseRow* row,
                           Model& model,
                           real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sum_w = 0;
  real_t *w = model.GetParameter_w();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    sum_w += iter->feat_val * w[iter->feat_id*3];
  }
  // bias
  w = model.GetParameter_b();
  sum_w += w[0];
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t align0 = 3 * model.get_aligned_k();
  index_t align1 = model.GetNumField() * align0;
  int align = kAlign * 3;
  w = model.GetParameter_v();
  __m128 XMMt = _mm_setzero_ps();
  for (SparseRow::const_iterator iter_i = row->begin();
       iter_i != row->end(); ++iter_i) {
    index_t j1 = iter_i->feat_id;
    index_t f1 = iter_i->field_id;
    real_t v1 = iter_i->feat_val;
    for (SparseRow::const_iterator iter_j = iter_i+1;
         iter_j != row->end(); ++iter_j) {
      index_t j2 = iter_j->feat_id;
      index_t f2 = iter_j->field_id;
      real_t v2 = iter_j->feat_val;
      real_t* w1_base = w + j1*align1 + f2*align0;
      real_t* w2_base = w + j2*align1 + f1*align0;
      __m128 XMMv = _mm_set1_ps(v1*v2);
      for (index_t d = 0; d < align0; d += align) {
        __m128 XMMw1 = _mm_load_ps(w1_base + d);
        __m128 XMMw2 = _mm_load_ps(w2_base + d);
        XMMt = _mm_add_ps(XMMt,
               _mm_mul_ps(
               _mm_mul_ps(XMMw1, XMMw2), XMMv));
      }
    }
  }
  real_t sum_v = 0;
  XMMt = _mm_hadd_ps(XMMt, XMMt);
  XMMt = _mm_hadd_ps(XMMt, XMMt);
  _mm_store_ss(&sum_v, XMMt);

  return sum_v + sum_w;
}

// Calculate gradient and update current model.
// Using the SSE to accelerate vector operation.
void FFMScoreFtrl::CalcGrad(const SparseRow* row,
                        Model& model,
                        real_t pg,
                        real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t *w = model.GetParameter_w();
  real_t alpha = 0.05;
  real_t beta = 1.0;
  real_t lambda1 = 0.0;
  real_t lambda2 = 0.0;

  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    real_t gradient = pg * iter->feat_val;
    index_t idx_w = iter->feat_id * 3;
    index_t idx_n = iter->feat_id * 3+1;
    index_t idx_z = iter->feat_id * 3+2;
    real_t old_n = w[idx_n];
    w[idx_n] += gradient * gradient;
    real_t sigma = 1.0f
                   * (std::sqrt(w[idx_n]) - std::sqrt(old_n))
                   / alpha;
    w[idx_z] += gradient - sigma * w[idx_w];
    if (std::abs(w[idx_z]) <= lambda1) {
      w[idx_w] = 0.0;
    } else {
      real_t smooth_lr = 1.0f
                         / (lambda2 + (beta + std::sqrt(w[idx_n])) / alpha);
      if (w[idx_z] < 0.0) {
        w[idx_z] += lambda1;
      } else if (w[idx_z] > 0.0) {
        w[idx_z] -= lambda1;
      }
      w[idx_w] = -1.0f * smooth_lr * w[idx_z];
    }
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t &wbn = w[1];
  real_t &wbz = w[2];
  real_t g = -1.0 * pg;
  wbn += g*g;
  if (std::abs(wbz) <= lambda1) {
    wb = 0.0f;
  } else {
    real_t smooth_lr = 1.0f
      / (lambda2 + (beta + std::sqrt(wbn)) / alpha);
    if (wbz < 0.0) {
      wbz += lambda1;
    } else if (wbz > 0.0) {
      wbz -= lambda1;
    }
    wb = -1.0f * smooth_lr * wbz;
  }

  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t align0 = 3 * model.get_aligned_k();
  index_t align1 = model.GetNumField() * align0;
  index_t align = kAlign * 3;
  w = model.GetParameter_v();
  __m128 XMMpg = _mm_set1_ps(pg);
  __m128 XMMcoef = _mm_set1_ps(-1.0);
  __m128 XMMalpha = _mm_set1_ps(alpha);
  __m128 XMMbeta = _mm_set1_ps(beta);
  __m128 XMMlambda1 = _mm_set1_ps(lambda1);
  __m128 XMMlambda2 = _mm_set1_ps(lambda2);
  __m128 XMMzero = _mm_set1_ps(0.0);
  for (SparseRow::const_iterator iter_i = row->begin();
       iter_i != row->end(); ++iter_i) {
    index_t j1 = iter_i->feat_id;
    index_t f1 = iter_i->field_id;
    real_t v1 = iter_i->feat_val;
    for (SparseRow::const_iterator iter_j = iter_i+1;
         iter_j != row->end(); ++iter_j) {
      index_t j2 = iter_j->feat_id;
      index_t f2 = iter_j->field_id;
      real_t v2 = iter_j->feat_val;
      real_t* w1_base = w + j1*align1 + f2*align0;
      real_t* w2_base = w + j2*align1 + f1*align0;
      __m128 XMMv = _mm_set1_ps(v1*v2);
      __m128 XMMpgv = _mm_mul_ps(XMMv, XMMpg);
      for (index_t d = 0; d < align0; d += align) {
        real_t *w1 = w1_base + d;
        real_t *w2 = w2_base + d;
        real_t *n1 = w1 + kAlign;
        real_t *n2 = w2 + kAlign;
        real_t *z1 = w1 + kAlign * 2;
        real_t *z2 = w2 + kAlign * 2;
        __m128 XMMw1 = _mm_load_ps(w1);
        __m128 XMMw2 = _mm_load_ps(w2);
        __m128 XMMn1 = _mm_load_ps(n1);
        __m128 XMMn2 = _mm_load_ps(n2);
        __m128 XMMz1 = _mm_load_ps(z1);
        __m128 XMMz2 = _mm_load_ps(z2);
        __m128 XMMg1 = _mm_mul_ps(XMMpgv, XMMw2);
        __m128 XMMg2 = _mm_mul_ps(XMMpgv, XMMw1);
        __m128 XMMold_n1 = XMMn1;
        __m128 XMMold_n2 = XMMn2;
        XMMn1 = _mm_add_ps(XMMn1, _mm_mul_ps(XMMg1, XMMg1));
        XMMn2 = _mm_add_ps(XMMn2, _mm_mul_ps(XMMg2, XMMg2));
        __m128 XMMsigma1 = _mm_mul_ps(XMMw1,
                           _mm_div_ps(
                           _mm_sub_ps(
                           _mm_rsqrt_ps(XMMn1),
                           _mm_rsqrt_ps(XMMold_n1)), XMMalpha));
        __m128 XMMsigma2 = _mm_mul_ps(XMMw2,
                           _mm_div_ps(
                           _mm_sub_ps(
                           _mm_rsqrt_ps(XMMn2),
                           _mm_rsqrt_ps(XMMold_n2)), XMMalpha));
        XMMz1 = _mm_sub_ps(
                _mm_add_ps(XMMz1,XMMg1),  XMMsigma1);
        XMMz2 = _mm_sub_ps(
                _mm_add_ps(XMMz2,XMMg2),  XMMsigma2);

        static const union {
          int i[4];
          __m128 m;
        } __mm_abs_mask_cheat_ps = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
        __m128 XMMcomp_res1 = _mm_cmplt_ps(XMMlambda1,
                              _mm_and_ps(XMMz1, __mm_abs_mask_cheat_ps.m));
        __m128 XMMcomp_res2 = _mm_cmplt_ps(XMMlambda2,
                              _mm_and_ps(XMMz2, __mm_abs_mask_cheat_ps.m));


        _mm_store_ps(comp_res1, XMMcomp_res1);
        _mm_store_ps(comp_res2, XMMcomp_res2);
        if (*comp_res1) {
          __m128 XMMsmooth_lr = _mm_rcp_ps(
                                _mm_add_ps(XMMlambda2,
                                _mm_div_ps(
                                _mm_add_ps(XMMbeta,
                                _mm_rsqrt_ps(XMMn1)),XMMalpha)));
          _mm_store_ps(comp_z_lt_zero, _mm_cmplt_ps(XMMz1, XMMzero));
          _mm_store_ps(comp_z_gt_zero, _mm_cmplt_ps(XMMzero, XMMz1));
          if (*comp_z_lt_zero) {
            XMMz1 = _mm_add_ps(XMMz1, XMMlambda1);
          } else if(*comp_z_gt_zero) {
            XMMz1 = _mm_sub_ps(XMMz1, XMMlambda1);
          }
          XMMw1 = _mm_mul_ps(XMMcoef,
                  _mm_mul_ps(XMMsmooth_lr, XMMz1));
        } else {
          XMMw1 = _mm_set1_ps(0.0);
        }

        if (*comp_res2) {
          __m128 XMMsmooth_lr = _mm_rcp_ps(
                                _mm_add_ps(XMMlambda2,
                                _mm_div_ps(
                                _mm_add_ps(XMMbeta,
                                _mm_rsqrt_ps(XMMn2)),XMMalpha)));
          _mm_store_ps(comp_z_lt_zero, _mm_cmplt_ps(XMMz2, XMMzero));
          _mm_store_ps(comp_z_gt_zero, _mm_cmplt_ps(XMMzero, XMMz2));
          if (*comp_z_lt_zero) {
            XMMz2 = _mm_add_ps(XMMz2, XMMlambda1);
          } else if (*comp_z_gt_zero) {
            XMMz2 = _mm_sub_ps(XMMz2, XMMlambda1);
          }
          XMMw2 = _mm_mul_ps(XMMcoef,
                  _mm_mul_ps(XMMsmooth_lr, XMMz2));
        } else {
          XMMw2 = _mm_set1_ps(0.0);
        }
        _mm_store_ps(w1, XMMw1);
        _mm_store_ps(w2, XMMw2);
        _mm_store_ps(n1, XMMn1);
        _mm_store_ps(n2, XMMn2);
        _mm_store_ps(z1, XMMz1);
        _mm_store_ps(z2, XMMz2);
      }
    }
  }
}

} // namespace xLearn