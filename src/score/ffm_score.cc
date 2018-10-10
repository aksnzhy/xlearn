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
This file is the implementation of FFMScore class.
*/

#include <pmmintrin.h>  // for SSE

#include "src/score/ffm_score.h"
#include "src/base/math.h"

namespace xLearn {

// y = sum( (V_i_fj*V_j_fi)(x_i * x_j) )
// Using SSE to accelerate vector operation.
// TODO(aksnzhy): solve unseen feature
real_t FFMScore::CalcScore(const SparseRow* row,
                           Model& model,
                           real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sum_w = 0;
  real_t sqrt_norm = sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  index_t num_field = model.GetNumField();
  index_t aux_size = model.GetAuxiliarySize();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t feat_id = iter->feat_id;
    // To avoid unseen feature in Prediction
    if (feat_id >= num_feat) continue;
    sum_w += (iter->feat_val * w[feat_id*aux_size] * sqrt_norm);
  }
  // bias
  w = model.GetParameter_b();
  sum_w += w[0];
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t align0 = aux_size * model.get_aligned_k();
  index_t align1 = num_field * align0;
  int align = kAlign * aux_size;
  w = model.GetParameter_v();
  __m128 XMMt = _mm_setzero_ps();
  for (SparseRow::const_iterator iter_i = row->begin();
       iter_i != row->end(); ++iter_i) {
    index_t j1 = iter_i->feat_id;
    index_t f1 = iter_i->field_id;
    // To avoid unseen feature in Prediction
    if (j1 >= num_feat || f1 >= num_field) continue;
    real_t v1 = iter_i->feat_val;
    for (SparseRow::const_iterator iter_j = iter_i+1;
         iter_j != row->end(); ++iter_j) {
      index_t j2 = iter_j->feat_id;
      index_t f2 = iter_j->field_id;
      // To avoid unseen feature in Prediction
      if (j2 >= num_feat || f2 >= num_field) continue;
      real_t v2 = iter_j->feat_val;
      real_t* w1_base = w + j1*align1 + f2*align0;
      real_t* w2_base = w + j2*align1 + f1*align0;
      __m128 XMMv = _mm_set1_ps(v1*v2*norm);
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
void FFMScore::CalcGrad(const SparseRow* row,
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
// TODO(aksnzhy): solve unseen feature
void FFMScore::calc_grad_sgd(const SparseRow* row,
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
    wl -= (learning_rate_ * g);
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t g = pg;
  wb -= (learning_rate_ * g);
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t align0 = model.GetAuxiliarySize() * model.get_aligned_k();
  index_t align1 = model.GetNumField() * align0;
  index_t align = kAlign * model.GetAuxiliarySize();
  w = model.GetParameter_v();
  __m128 XMMpg = _mm_set1_ps(pg);
  __m128 XMMlr = _mm_set1_ps(learning_rate_);
  __m128 XMMlamb = _mm_set1_ps(regu_lambda_);
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
      __m128 XMMv = _mm_set1_ps(v1*v2*norm);
      __m128 XMMpgv = _mm_mul_ps(XMMv, XMMpg);
      for (index_t d = 0; d < align0; d += align) {
        real_t *w1 = w1_base + d;
        real_t *w2 = w2_base + d;
        __m128 XMMw1 = _mm_load_ps(w1);
        __m128 XMMw2 = _mm_load_ps(w2);
        __m128 XMMg1 = _mm_add_ps(
                       _mm_mul_ps(XMMlamb, XMMw1),
                       _mm_mul_ps(XMMpgv, XMMw2));
        __m128 XMMg2 = _mm_add_ps(
                       _mm_mul_ps(XMMlamb, XMMw2),
                       _mm_mul_ps(XMMpgv, XMMw1));
        XMMw1 = _mm_sub_ps(XMMw1, _mm_mul_ps(XMMlr, XMMg1));
        XMMw2 = _mm_sub_ps(XMMw2, _mm_mul_ps(XMMlr, XMMg2));
        _mm_store_ps(w1, XMMw1);
        _mm_store_ps(w2, XMMw2);
      }
    }
  }

}

// Calculate gradient and update current model using adagrad
// TODO(aksnzhy): solve unseen feature
void FFMScore::calc_grad_adagrad(const SparseRow* row,
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
  index_t align0 = 2 * model.get_aligned_k();
  index_t align1 = model.GetNumField() * align0;
  index_t align = kAlign * 2;
  w = model.GetParameter_v();
  __m128 XMMpg = _mm_set1_ps(pg);
  __m128 XMMlr = _mm_set1_ps(learning_rate_);
  __m128 XMMlamb = _mm_set1_ps(regu_lambda_);
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
      __m128 XMMv = _mm_set1_ps(v1*v2*norm);
      __m128 XMMpgv = _mm_mul_ps(XMMv, XMMpg);
      for (index_t d = 0; d < align0; d += align) {
        real_t *w1 = w1_base + d;
        real_t *w2 = w2_base + d;
        real_t *wg1 = w1 + kAlign;
        real_t *wg2 = w2 + kAlign;
        __m128 XMMw1 = _mm_load_ps(w1);
        __m128 XMMw2 = _mm_load_ps(w2);
        __m128 XMMwg1 = _mm_load_ps(wg1);
        __m128 XMMwg2 = _mm_load_ps(wg2);
        __m128 XMMg1 = _mm_add_ps(
                       _mm_mul_ps(XMMlamb, XMMw1),
                       _mm_mul_ps(XMMpgv, XMMw2));
        __m128 XMMg2 = _mm_add_ps(
                       _mm_mul_ps(XMMlamb, XMMw2),
                       _mm_mul_ps(XMMpgv, XMMw1));
        XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
        XMMwg2 = _mm_add_ps(XMMwg2, _mm_mul_ps(XMMg2, XMMg2));
        XMMw1 = _mm_sub_ps(XMMw1, _mm_mul_ps(XMMlr,
                _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));
        XMMw2 = _mm_sub_ps(XMMw2, _mm_mul_ps(XMMlr,
                _mm_mul_ps(_mm_rsqrt_ps(XMMwg2), XMMg2)));
        _mm_store_ps(w1, XMMw1);
        _mm_store_ps(w2, XMMw2);
        _mm_store_ps(wg1, XMMwg1);
        _mm_store_ps(wg2, XMMwg2);
      }
    }
  }
}

// Calculate gradient and update current model using ftrl
// TODO(aksnzhy): solve unseen feature
void FFMScore::calc_grad_ftrl(const SparseRow* row,
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
  index_t align0 = 3 * model.get_aligned_k();
  index_t align1 = model.GetNumField() * align0;
  index_t align = kAlign * 3;
  w = model.GetParameter_v();
  __m128 XMMpg = _mm_set1_ps(pg);
  __m128 XMMalpha = _mm_set1_ps(alpha_);
  __m128 XMML2 = _mm_set1_ps(lambda_2_);
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
      __m128 XMMv = _mm_set1_ps(v1*v2*norm);
      __m128 XMMpgv = _mm_mul_ps(XMMv, XMMpg);
      for (index_t d = 0; d < align0; d += align) {
        real_t *w1 = w1_base + d;
        real_t *w2 = w2_base + d;
        real_t *wg1 = w1 + kAlign;
        real_t *wg2 = w2 + kAlign;
        real_t *z1 = w1 + kAlign * 2;
        real_t *z2 = w2 + kAlign * 2;
        __m128 XMMw1 = _mm_load_ps(w1);
        __m128 XMMw2 = _mm_load_ps(w2);
        __m128 XMMwg1 = _mm_load_ps(wg1);
        __m128 XMMwg2 = _mm_load_ps(wg2);
        __m128 XMMz1 = _mm_load_ps(z1);
        __m128 XMMz2 = _mm_load_ps(z2);
        __m128 XMMg1 = _mm_add_ps(
                       _mm_mul_ps(XMML2, XMMw1),
                       _mm_mul_ps(XMMpgv, XMMw2));
        __m128 XMMg2 = _mm_add_ps(
                       _mm_mul_ps(XMML2, XMMw2),
                       _mm_mul_ps(XMMpgv, XMMw1));
        __m128 XMMsigma1 = _mm_div_ps(
                           _mm_sub_ps(
                           _mm_sqrt_ps(
                           _mm_add_ps(XMMwg1,
                           _mm_mul_ps(XMMg1, XMMg1))),
                           _mm_sqrt_ps(XMMwg1)), XMMalpha);
        __m128 XMMsigma2 = _mm_div_ps(
                           _mm_sub_ps(
                           _mm_sqrt_ps(
                           _mm_add_ps(XMMwg2,
                           _mm_mul_ps(XMMg2, XMMg2))),
                           _mm_sqrt_ps(XMMwg2)), XMMalpha);
        XMMz1 = _mm_add_ps(XMMz1,
                _mm_sub_ps(XMMg1,
                _mm_mul_ps(XMMsigma1, XMMw1)));
        XMMz2 = _mm_add_ps(XMMz2,
                _mm_sub_ps(XMMg2,
                _mm_mul_ps(XMMsigma2, XMMw2)));
        _mm_store_ps(z1, XMMz1);
        _mm_store_ps(z2, XMMz2);
        XMMwg1 = _mm_add_ps(XMMwg1,
                 _mm_mul_ps(XMMg1, XMMg1));
        XMMwg2 = _mm_add_ps(XMMwg2,
                 _mm_mul_ps(XMMg2, XMMg2));
        _mm_store_ps(wg1, XMMwg1);
        _mm_store_ps(wg2, XMMwg2);
        // Update w. SSE may not faster
        for (size_t i = 0; i < kAlign; ++i) {
          // w1
          real_t z1_value = *(z1+i);
          int sign = z1_value > 0 ? 1 : -1;
          if (sign * z1_value <= lambda_1_) {
            *(w1+i) = 0;
          } else {
            *(w1+i) = (sign*lambda_1_-z1_value) / 
              ((beta_ + sqrt(*(wg1+i))) / alpha_ + lambda_2_);
          }
          // w2
          real_t z2_value = *(z2+i);
          sign = z2_value > 0 ? 1 : -1;
          if (sign * z2_value <= lambda_1_) {
            *(w2+i) = 0;
          } else {
            *(w2+i) = (sign*lambda_1_-z2_value) / 
              ((beta_ + sqrt(*(wg2+i))) / alpha_ + lambda_2_);
          }
        }
      }
    }
  }
}

} // namespace xLearn
