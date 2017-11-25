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

#include <pmmintrin.h>  // for SSE

#include "src/score/fm_score_ftrl.h"
#include "src/base/math.h"

namespace xLearn {

// y = sum( (V_i*V_j)(x_i * x_j) )
// Using SSE to accelerate vector operation.
real_t FMScoreFtrl::CalcScore(const SparseRow* row,
                          Model& model,
                          real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = sqrt(norm);
  real_t *w = model.GetParameter_w();
  real_t t = 0;
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    t += (iter->feat_val * w[iter->feat_id*3] * sqrt_norm);
  }
  // bias
  w = model.GetParameter_b();
  t += w[0];
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t aligned_k = model.get_aligned_k();
  index_t align0 = model.get_aligned_k() * 3;
  std::vector<real_t> sv(aligned_k, 0);
  real_t* s = sv.data();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t j1 = iter->feat_id;
    real_t v1 = iter->feat_val;
    real_t *w = model.GetParameter_v() + j1 * align0;
    __m128 XMMv = _mm_set1_ps(v1*norm);
    for (index_t d = 0; d < aligned_k; d += kAlign) {
      __m128 XMMs = _mm_load_ps(s+d);
      __m128 const XMMw = _mm_load_ps(w+d);
      XMMs = _mm_add_ps(XMMs, _mm_mul_ps(XMMw, XMMv));
      _mm_store_ps(s+d, XMMs);
    }
  }
  __m128 XMMt = _mm_set1_ps(0.0f);
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t j1 = iter->feat_id;
    real_t v1 = iter->feat_val;
    real_t *w = model.GetParameter_v() + j1 * align0;
    __m128 XMMv = _mm_set1_ps(v1*norm);
    for (index_t d = 0; d < aligned_k; d += kAlign) {
      __m128 XMMs = _mm_load_ps(s+d);
      __m128 XMMw = _mm_load_ps(w+d);
      __m128 XMMwv = _mm_mul_ps(XMMw, XMMv);
      XMMt = _mm_add_ps(XMMt,
             _mm_mul_ps(XMMwv, _mm_sub_ps(XMMs, XMMwv)));
    }
  }
  XMMt = _mm_hadd_ps(XMMt, XMMt);
  XMMt = _mm_hadd_ps(XMMt, XMMt);
  real_t t_all;
  _mm_store_ss(&t_all, XMMt);
  t_all *= 0.5;
  t_all += t;
  return t_all;
}

// Calculate gradient and update current model parameters.
// Using SSE to accelerate vector operation.
void FMScoreFtrl::CalcGrad(const SparseRow* row,
                       Model& model,
                       real_t pg,
                       real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = sqrt(norm);
  real_t *w = model.GetParameter_w();
  real_t alpha = .01;
  real_t beta = 1.0;
  real_t lambda1 = 2.0;
  real_t lambda2 = 4.0;
  for (SparseRow::const_iterator iter = row->begin();
      iter != row->end(); ++iter) {
    real_t gradient = pg * iter->feat_val;
    index_t idx_w = iter->feat_id * 3;
    index_t idx_n = idx_w + 1;
    index_t idx_z = idx_w + 2;
    real_t old_n =  w[idx_n];
    w[idx_n] += gradient * gradient;
    real_t sigma = 1.0f
                   * (std::sqrt(w[idx_n]) - std::sqrt(old_n))
                   / alpha;
    w[idx_z] += gradient - sigma * w[idx_w];

    if (std::abs(w[idx_z]) <= lambda1) {
      w[idx_w] = 0;
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
  real_t g = pg; // g = pg * iter->feat_val; ?
  wbn += g*g;
  wbz += g;
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
  index_t aligned_k = model.get_aligned_k();
  index_t align0 = model.get_aligned_k() * 3;
  __m128 XMMpg = _mm_set1_ps(pg);
  std::vector<real_t> sv(aligned_k, 0);
  real_t* s = sv.data();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t j1 = iter->feat_id;
    real_t v1 = iter->feat_val;
    real_t *w = model.GetParameter_v() + j1 * align0;
    __m128 XMMv = _mm_set1_ps(v1*norm);
    for (index_t d = 0; d < aligned_k; d += kAlign) {
      __m128 XMMs = _mm_load_ps(s+d);
      __m128 const XMMw = _mm_load_ps(w+d);
      XMMs = _mm_add_ps(XMMs, _mm_mul_ps(XMMw, XMMv));
      _mm_store_ps(s+d, XMMs);
    }
  }
  __m128 XMMcoef = _mm_set1_ps(-1.0);
  __m128 XMMalpha = _mm_set1_ps(alpha);
  __m128 XMMbeta = _mm_set1_ps(beta);
  __m128 XMMlambda1 = _mm_set1_ps(lambda1);
  __m128 XMMlambda2 = _mm_set1_ps(lambda2);
  __m128 XMMzero = _mm_set1_ps(0.0);
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t j1 = iter->feat_id;
    real_t v1 = iter->feat_val;
    real_t *w = model.GetParameter_v() + j1 * align0;
    __m128 XMMv = _mm_set1_ps(v1*norm);
    __m128 XMMpgv = _mm_mul_ps(XMMpg, XMMv);
    for(index_t d = 0; d < aligned_k; d += kAlign) {
      __m128 XMMs = _mm_load_ps(s+d);
      __m128 XMMw = _mm_load_ps(w+d);
      __m128 XMMn = _mm_load_ps(w+aligned_k+d);
      __m128 XMMz = _mm_load_ps(w+aligned_k*2+d);
      __m128 XMMg = _mm_mul_ps(XMMpgv,
                    _mm_sub_ps(XMMs,
                    _mm_mul_ps(XMMw, XMMv)));
      __m128 XMMold_n = XMMn;
      XMMn = _mm_add_ps(XMMn,
             _mm_mul_ps(XMMg, XMMg));
      __m128 XMMsigma = _mm_mul_ps(
                        _mm_div_ps(
                        _mm_sub_ps(_mm_rsqrt_ps(XMMn),
                        _mm_rsqrt_ps(XMMold_n)), XMMalpha), XMMw);
      XMMz = _mm_sub_ps(
             _mm_add_ps(XMMz, XMMg), XMMsigma);
      static const union {
        int i[4];
        __m128 m;
      } __mm_abs_mask_cheat_ps = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
      __m128 XMMcomp_res = _mm_cmplt_ps(XMMlambda1,
                           _mm_and_ps(XMMz,
                          __mm_abs_mask_cheat_ps.m));
      real_t* comp_res;
      _mm_store_ps(comp_res, XMMcomp_res);
      if (comp_res) {
        __m128 XMMsmooth_lr = _mm_rcp_ps(
                              _mm_add_ps(XMMlambda2,
                              _mm_div_ps(
                              _mm_add_ps(XMMbeta,
                              _mm_rsqrt_ps(XMMn)),XMMalpha)));
        real_t* comp_z_lt_zero;
        _mm_store_ps(comp_z_lt_zero, _mm_cmplt_ps(XMMz, XMMzero));
        real_t* comp_z_gt_zero;
        _mm_store_ps(comp_z_gt_zero, _mm_cmpgt_ps(XMMzero, XMMz));
        if (comp_z_lt_zero) {
          XMMz = _mm_add_ps(XMMz, XMMlambda1);
        } else if(comp_z_gt_zero) {
          XMMz = _mm_sub_ps(XMMz, XMMlambda1);
        }
        XMMw = _mm_mul_ps(XMMcoef,
                          _mm_mul_ps(XMMsmooth_lr, XMMz));
      } else {
        XMMw = _mm_set1_ps(0.0);
      }
      _mm_store_ps(w+d, XMMw);
      _mm_store_ps(w+aligned_k+d, XMMn);
      _mm_store_ps(w+aligned_k*2+d, XMMz);
    }
  }
}

} // namespace xLearn
