//------------------------------------------------------------------------------
// Copyright (c) 2018 by contributors. All Rights Reserved.
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
This file is the implementation of FMScore class.
*/

#include <pmmintrin.h>  // for SSE

#include "src/score/fm_score.h"
#include "src/base/math.h"

namespace xLearn {

// y = sum( (V_i*V_j)(x_i * x_j) )
// Using SSE to accelerate vector operation.
// TODO(aksnzhy): solve unseen feature
real_t FMScore::CalcScore(const SparseRow* row,
                          Model& model,
                          real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  real_t t = 0;
  index_t aux_size = model.GetAuxiliarySize();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t feat_id = iter->feat_id;
    // To avoid unseen feature in Prediction
    if (feat_id >= num_feat) continue;
    t += (iter->feat_val * w[feat_id*aux_size] * sqrt_norm);
  }
  // bias
  w = model.GetParameter_b();
  t += w[0];
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t aligned_k = model.get_aligned_k();
  index_t align0 = model.get_aligned_k() * aux_size;
  std::vector<real_t> sv(aligned_k, 0);
  real_t* s = sv.data();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t j1 = iter->feat_id;
    // To avoid unseen feature in Prediction
    if (j1 >= num_feat) continue;
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
    // To avoid unseen feature in Prediction
    if (j1 >= num_feat) continue;
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
// TODO(aksnzhy): solve unseen feature
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
  else {
    LOG(FATAL) << "Unknow optimization method: " << opt_type_;
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
  index_t align0 = model.get_aligned_k() * 
                   model.GetAuxiliarySize();
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
// TODO(aksnzhy): solve unseen feature
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
  index_t align0 = model.get_aligned_k() * 
                   model.GetAuxiliarySize();
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
// TODO(aksnzhy): solve unseen feature
void FMScore::calc_grad_ftrl(const SparseRow* row,
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
    real_t &wl = w[iter->feat_id*3];
    real_t &wlg = w[iter->feat_id*3+1];
    real_t &wlz = w[iter->feat_id*3+2];
    real_t g = lambda_2_*wl+pg*iter->feat_val*sqrt_norm; 
    real_t old_wlg = wlg;
    wlg += g*g;
    real_t sigma = (sqrt(wlg)-sqrt(old_wlg)) / alpha_;
    wlz += (g-sigma*wl);
    int sign = wlz > 0 ? 1:-1;
    if (sign*wlz <= lambda_1_) {
      wl = 0;
    } else {
      wl = (sign*lambda_1_-wlz) / 
           ((beta_ + sqrt(wlg)) / 
            alpha_ + lambda_2_);
    }
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t &wbg = w[1];
  real_t &wbz = w[2];
  real_t g = pg;
  real_t old_wbg = wbg;
  wbg += g*g;
  real_t sigma = (sqrt(wbg)-sqrt(old_wbg)) / alpha_;
  wbz += (g-sigma*wb);
  int sign = wbz > 0 ? 1:-1;
  if (sign*wbz <= lambda_1_) {
    wb = 0;
  } else {
    wb = (sign*lambda_1_-wbz) / 
         ((beta_ + sqrt(wbg)) / 
          alpha_ + lambda_2_);
  }  
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t aligned_k = model.get_aligned_k();
  index_t align0 = model.get_aligned_k() * 
                   model.GetAuxiliarySize();
  __m128 XMMpg = _mm_set1_ps(pg);
  __m128 XMMalpha = _mm_set1_ps(alpha_);
  __m128 XMML2 = _mm_set1_ps(lambda_2_);
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
    real_t *w_base = model.GetParameter_v() + j1 * align0;
    __m128 XMMv = _mm_set1_ps(v1*norm);
    __m128 XMMpgv = _mm_mul_ps(XMMpg, XMMv);
    for (index_t d = 0; d < aligned_k; d += kAlign) {
      real_t* w = w_base + d;
      real_t* wg = w_base + aligned_k + d;
      real_t* z = w_base + aligned_k*2 + d;
      __m128 XMMs = _mm_load_ps(s+d);
      __m128 XMMw = _mm_load_ps(w);
      __m128 XMMwg = _mm_load_ps(wg);
      __m128 XMMz = _mm_load_ps(z);
      __m128 XMMg = _mm_add_ps(
                    _mm_mul_ps(XMML2, XMMw),
                    _mm_mul_ps(XMMpgv,
                    _mm_sub_ps(XMMs,
                      _mm_mul_ps(XMMw, XMMv))));
      __m128 XMMsigma = _mm_div_ps(
                        _mm_sub_ps(
                        _mm_sqrt_ps(
                        _mm_add_ps(XMMwg,
                        _mm_mul_ps(XMMg, XMMg))),
                        _mm_sqrt_ps(XMMwg)), XMMalpha);
      XMMz = _mm_add_ps(XMMz,
             _mm_sub_ps(XMMg,
             _mm_mul_ps(XMMsigma, XMMw)));
      _mm_store_ps(z, XMMz);
      XMMwg = _mm_add_ps(XMMwg,
              _mm_mul_ps(XMMg, XMMg));
      _mm_store_ps(wg, XMMwg);
      // Update w. SSE mat not faster.
      for (size_t i = 0; i < kAlign; ++i) {
        real_t z_value = *(z+i);
        int sign = z_value > 0 ? 1:-1;
        if (sign * z_value <= lambda_1_) {
          *(w+i) = 0;
        } else {
          *(w+i) = (sign*lambda_1_-z_value) / 
              ((beta_ + sqrt(*(wg+i))) / alpha_ + lambda_2_);
        }
      }
    }
  }
}

} // namespace xLearn
