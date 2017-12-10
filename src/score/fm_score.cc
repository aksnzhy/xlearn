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

#include "src/score/fm_score.h"
#include "src/base/math.h"

namespace xLearn {

// y = sum( (V_i*V_j)(x_i * x_j) )
// Using SSE to accelerate vector operation.
real_t FMScore::CalcScore(const SparseRow* row,
                          Model& model,
                          real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = sqrt(norm);
  real_t *w = model.GetParameter_w();
  real_t t = 0;
  index_t auxiliary_size = model.GetAuxiliarySize();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    t += (iter->feat_val * w[iter->feat_id*auxiliary_size] * sqrt_norm);
  }
  // bias
  w = model.GetParameter_b();
  t += w[0];
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t aligned_k = model.get_aligned_k();
  index_t align0 = model.get_aligned_k() * model.GetAuxiliarySize();
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
void FMScore::CalcGrad(const SparseRow* row,
                       Model& model,
                       real_t pg,
                       real_t norm) {
  // Using sgd
  if (opt_type_.compare("sgd") == 0) {
    this->calc_grad_sgd(row, model, pg, norm);
  }
  // Using adagrad
  else if (opt_type_.compare("adagrad") == 0) {
    this->calc_grad_adagrad(row, model, pg, norm);
  }
  // Using ftrl 
  else if (opt_type_.compare("ftrl") == 0) {
    this->calc_grad_ftrl(row, model, pg, norm);
  }
}

// Calculate gradient and update current model using sgd
void FMScore::calc_grad_sgd(const SparseRow* row,
                            Model& model,
                            real_t pg,
                            real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/  
  real_t sqrt_norm = sqrt(norm);
  real_t *w = model.GetParameter_w();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    real_t &wl = w[iter->feat_id];
    real_t g = regu_lambda_*wl+pg*iter->feat_val*sqrt_norm;
    wl -= learning_rate_ * g;
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t g = pg;
  wb -= learning_rate_ * g;
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t aligned_k = model.get_aligned_k();
  index_t align0 = model.get_aligned_k() * model.GetAuxiliarySize();
  __m128 XMMpg = _mm_set1_ps(pg);
  __m128 XMMlr = _mm_set1_ps(learning_rate_);
  __m128 XMMlamb = _mm_set1_ps(regu_lambda_);
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
      __m128 XMMg = _mm_add_ps(_mm_mul_ps(XMMlamb, XMMw),
        _mm_mul_ps(XMMpgv, _mm_sub_ps(XMMs,
        _mm_mul_ps(XMMw, XMMv))));
      XMMw = _mm_sub_ps(XMMw, _mm_mul_ps(XMMlr, XMMg));
      _mm_store_ps(w+d, XMMw);
    }
  }
}

// Calculate gradient and update current model using adagrad
void FMScore::calc_grad_adagrad(const SparseRow* row,
                                Model& model,
                                real_t pg,
                                real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = sqrt(norm);
  real_t *w = model.GetParameter_w();
  for (SparseRow::const_iterator iter = row->begin();
      iter != row->end(); ++iter) {
    real_t &wl = w[iter->feat_id*2];
    real_t &wlg = w[iter->feat_id*2+1];
    real_t g = regu_lambda_*wl+pg*iter->feat_val*sqrt_norm;
    wlg += g*g;
    wl -= learning_rate_ * g * InvSqrt(wlg);
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t &wbg = w[1];
  real_t g = pg;
  wbg += g*g;
  wb -= learning_rate_ * g * InvSqrt(wbg);
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t aligned_k = model.get_aligned_k();
  index_t align0 = model.get_aligned_k() * model.GetAuxiliarySize();
  __m128 XMMpg = _mm_set1_ps(pg);
  __m128 XMMlr = _mm_set1_ps(learning_rate_);
  __m128 XMMlamb = _mm_set1_ps(regu_lambda_);
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
      __m128 XMMwg = _mm_load_ps(w+aligned_k+d);
      __m128 XMMg = _mm_add_ps(_mm_mul_ps(XMMlamb, XMMw),
      _mm_mul_ps(XMMpgv, _mm_sub_ps(XMMs,
        _mm_mul_ps(XMMw, XMMv))));
      XMMwg = _mm_add_ps(XMMwg, _mm_mul_ps(XMMg, XMMg));
      XMMw = _mm_sub_ps(XMMw,
             _mm_mul_ps(XMMlr,
             _mm_mul_ps(_mm_rsqrt_ps(XMMwg), XMMg)));
      _mm_store_ps(w+d, XMMw);
      _mm_store_ps(w+aligned_k+d, XMMwg);
    }
  }
}

// Calculate gradient and update current model using ftrl
void FMScore::calc_grad_ftrl(const SparseRow* row,
                             Model& model,
                             real_t pg,
                             real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t *w = model.GetParameter_w();
  for (SparseRow::const_iterator iter = row->begin();
    iter != row->end(); ++iter) {
    real_t gradient = pg * iter->feat_val;
    index_t idx_w = iter->feat_id * 3;
    index_t idx_n = idx_w + 1;
    index_t idx_z = idx_w + 2;
    real_t old_n =  w[idx_n];
    w[idx_n] += gradient * gradient;
    real_t sqrt_n = sqrt(w[idx_n]);
    real_t sigma = (sqrt_n - sqrt(old_n)) / alpha_;
    w[idx_z] += gradient - sigma * w[idx_w];
    if (std::abs(w[idx_z]) <= lambda_1_) {
      w[idx_w] = 0;
    } else {
      real_t smooth_lr = -1.0f / (lambda_2_ + (beta_ + sqrt_n) / alpha_);
      if (w[idx_z] > 0.0) {
        w[idx_z] -= lambda_1_;
      }
      if (w[idx_z] < 0.0) {
        w[idx_z] += lambda_1_;
      }
      w[idx_w] = smooth_lr * w[idx_z];
    }
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t &wbn = w[1];
  real_t &wbz = w[2];
  real_t g = pg;
  real_t old_wbn = wbn;
  wbn += g*g;
  real_t sqrt_wbn = sqrt(wbn);
  real_t sigma_wb = (sqrt_wbn - sqrt(old_wbn)) / alpha_;
  wbz += g - sigma_wb * wb;
  if (std::abs(wbz) <= lambda_1_) {
    wb = 0.0f;
  } else {
    real_t smooth_lr = -1.0f
                       / (lambda_2_ + (beta_ + sqrt(wbn)) / alpha_);
    if (wbz > 0.0) {
      wbz -= lambda_1_;
    }
    if (wbz < 0.0) {
      wbz += lambda_1_;
    }
    wb = smooth_lr * wbz;
  }
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t aligned_k = model.get_aligned_k();
  index_t align0 = model.get_aligned_k() * model.GetAuxiliarySize();
  __m128 XMMpg = _mm_set1_ps(pg);
  std::vector<real_t> sv(aligned_k, 0);
  real_t* s = sv.data();
  for (SparseRow::const_iterator iter = row->begin();
      iter != row->end(); ++iter) {
    index_t j1 = iter->feat_id;
    real_t v1 = iter->feat_val;
    real_t *w = model.GetParameter_v() + j1 * align0;
    __m128 XMMv = _mm_set1_ps(v1);
    for (index_t d = 0; d < aligned_k; d += kAlign) {
      __m128 XMMs = _mm_load_ps(s+d);
      __m128 const XMMw = _mm_load_ps(w+d);
      XMMs = _mm_add_ps(XMMs, _mm_mul_ps(XMMw, XMMv));
      _mm_store_ps(s+d, XMMs);
    }
  }
  __m128 XMMcoef = _mm_set1_ps(-1.0);
  __m128 XMMalpha = _mm_set1_ps(alpha_);
  __m128 XMMbeta = _mm_set1_ps(beta_);
  __m128 XMMlambda1 = _mm_set1_ps(lambda_1_);
  __m128 XMMlambda2 = _mm_set1_ps(lambda_2_);
  __m128 XMMzero = _mm_set1_ps(0.0);
  if (comp_res == nullptr) {
    int ret = posix_memalign(
        (void**)&comp_res,
        kAlignByte,
        1 * sizeof(real_t));
    CHECK_EQ(ret, 0);
  }
  if (comp_z_lt_zero == nullptr) {
    int ret = posix_memalign(
        (void**)&comp_z_lt_zero,
        kAlignByte,
        1 * sizeof(real_t));
    CHECK_EQ(ret, 0);
  }
  if (comp_z_gt_zero == nullptr) {
    int ret = posix_memalign(
        (void**)&comp_z_gt_zero,
        kAlignByte,
        1 * sizeof(real_t));
    CHECK_EQ(ret, 0);
  }
  for (SparseRow::const_iterator iter = row->begin();
    iter != row->end(); ++iter) {
    index_t j1 = iter->feat_id;
    real_t v1 = iter->feat_val;
    real_t *w = model.GetParameter_v() + j1 * align0;
    __m128 XMMv = _mm_set1_ps(v1);
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
                        _mm_sub_ps(_mm_sqrt_ps(XMMn),
                        _mm_sqrt_ps(XMMold_n)), XMMalpha), XMMw);
      XMMz = _mm_sub_ps(
             _mm_add_ps(XMMz, XMMg), XMMsigma);
      static const union {
        int i[4];
        __m128 m;
      } __mm_abs_mask_cheat_ps = {{ 0x7fffffff, 0x7fffffff, 
                                    0x7fffffff, 0x7fffffff }};
      __m128 XMMcomp_res = _mm_cmplt_ps(XMMlambda1,
                           _mm_and_ps(XMMz,
                           __mm_abs_mask_cheat_ps.m));
      _mm_store_ps(comp_res, XMMcomp_res);
      if (comp_res) {
        __m128 XMMsmooth_lr = _mm_rcp_ps(
                              _mm_add_ps(XMMlambda2,
                              _mm_div_ps(
                              _mm_add_ps(XMMbeta,
                              _mm_sqrt_ps(XMMn)), XMMalpha)));
        _mm_store_ps(comp_z_lt_zero, _mm_cmplt_ps(XMMz, XMMzero));
        _mm_store_ps(comp_z_gt_zero, _mm_cmpgt_ps(XMMz, XMMzero));
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
