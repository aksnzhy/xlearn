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

#include "src/score/ffm_score.h"
#include "src/base/math.h"

namespace xLearn {

// y = sum( (V_i_fj*V_j_fi)(x_i * x_j) )
// Using SSE to accelerate vector operation.
  real_t DistFFMScore::CalcScore(const SparseRow* row,
      Model& model,
      std::map<index_t, real_t>* weight,
      std::map<index_t, std::vector<real_t>>* v,
      real_t norm = 1.0
      ) {
    /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sum_w = 0;
  real_t sqrt_norm = sqrt(norm);
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    sum_w += (iter->feat_val * 
              weight[iter->feat_id] * 
              sqrt_norm);
  }
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t align0 = aux_size * model.get_aligned_k();
  index_t align1 = model.GetNumField() * align0;
  int align = kAlign * aux_size;
  //w = model.GetParameter_v();
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
      real_t* w1_base = v[j1].data();
      real_t* w2_base = v[j2].data();
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
void DistFFMScore::CalcGrad(const DMatrix* matrix,
    Model& model,
    std::map<index_t, real_t>& w,
    std::map<index_t, std::vector<real_t>>& v,
    real_t* sum,
    std::map<index_t, real_t>& w_g,
    std::map<index_t, real_t>& v_g,
    index_t start_idx,
    index_t end_idx
    ) {
  // Using sgd
  if (opt_type_.compare("sgd") == 0) {
    this->calc_grad_sgd(matrix, model, w, v, sum, w_g, v_g, start_idx, end_idx);
  }
  // Using adagrad
  else if (opt_type_.compare("adagrad") == 0) {
    this->calc_grad_adagrad(matrix, model, w, v, sum, w_g, v_g, start_idx, end_idx);
  }
  // Using ftrl 
  else if (opt_type_.compare("ftrl") == 0) {
    this->calc_grad_ftrl(matrix, model, w, v, sum, w_g, v_g, start_idx, end_idx);
  }
}

// Calculate gradient and update current model using sgd
void DistFFMScore::calc_grad_sgd(const DMatrix* matrix,
    Model& model,
    std::map<index_t, real_t>& weight,
    std::map<index_t, std::vector<real_t>>& v,
    real_t* sum,
    std::map<index_t, real_t>& w_g,
    std::map<index_t, std::vector<real_t>>& v_g,
    index_t start_idx,
    index_t end_idx
    ) {
  for (index_t i = start_idx; i < end_idx; ++i) {
    SparseRow* row = matrix->row[i];
    real_t pred = CalcScore(row, model, &w, &v);
    real_t y = matrix->Y[i] > 0 ? 1.0 : -1.0;
    (*sum) += log1p(exp(-y*pred));
    real_t pg = -y / (1.0 + (1.0 / exp(-y * pred)));
    /*********************************************************
     *  linear term and bias term                            *
     *********************************************************/
    for (SparseRow::const_iterator iter = row->begin();
        iter != row->end(); ++iter) {
      real_t &wl = weight[iter->feat_id];
      real_t g = pg*iter->feat_val;
      w_g[iter->feat_id] += g;
    }
    /*********************************************************
     *  latent factor                                        *
     *********************************************************/
    index_t align0 = model.GetAuxiliarySize() * model.get_aligned_k();
    index_t align1 = model.GetNumField() * align0;
    index_t align = kAlign * model.GetAuxiliarySize();
    //w = model.GetParameter_v();
    __m128 XMMpg = _mm_set1_ps(pg);
    __m128 XMMlr = _mm_set1_ps(learning_rate_);
    __m128 XMMlamb = _mm_set1_ps(regu_lambda_);
    for (SparseRow::const_iterator iter_i = row->begin();
        iter_i != row->end(); ++iter_i) {
      index_t j1 = iter_i->feat_id;
      index_t f1 = iter_i->field_id;
      real_t v1 = iter_i->feat_val;
      real_t* w1_base = v[j1].data() + j1 * align1;
      for (SparseRow::const_iterator iter_j = iter_i+1;
          iter_j != row->end(); ++iter_j) {
        index_t j2 = iter_j->feat_id;
        index_t f2 = iter_j->field_id;
        real_t v2 = iter_j->feat_val;
        real_t* w2_base = v[j2].data() + j2 * align1;
        __m128 XMMv = _mm_set1_ps(v1*v2);
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
}

// Calculate gradient and update current model using adagrad
void DistFFMScore::calc_grad_adagrad(const DMatrix* matrix,
    Model& model,
    std::map<index_t, real_t>& weight,
    std::map<index_t, std::vector<real_t>>& v,
    real_t* sum,
    std::map<index_t, real_t>& w_g,
    std::map<index_t, std::vector<real_t>>& v_g,
    index_t start_idx,
    index_t end_idx
    ) {
  for (index_t i = start_idx; i < end_idx; ++i) {
    SparseRow* row = matrix->row[i];
    real_t pred = CalcScore(row, model, &w, &v);
    real_t y = matrix->Y[i] > 0 ? 1.0 : -1.0;
    (*sum) += log1p(exp(-y*pred));
    real_t pg = -y / (1.0 + (1.0 / exp(-y * pred)));
    /*********************************************************
     *  linear term and bias term                            *
     *********************************************************/
    for (SparseRow::const_iterator iter = row->begin();
        iter != row->end(); ++iter) {
      real_t &wl = weight[iter->feat_id];
      real_t g = pg*iter->feat_val;
      w_g[iter->feat_id] += g;
    }
    /*********************************************************
     *  latent factor                                        *
     *********************************************************/
    index_t align0 = model.get_aligned_k();
    index_t align = kAlign;
    //w = model.GetParameter_v();
    __m128 XMMpg = _mm_set1_ps(pg);
    __m128 XMMlr = _mm_set1_ps(learning_rate_);
    __m128 XMMlamb = _mm_set1_ps(regu_lambda_);
    for (SparseRow::const_iterator iter_i = row->begin();
        iter_i != row->end(); ++iter_i) {
      index_t j1 = iter_i->feat_id;
      index_t f1 = iter_i->field_id;
      real_t* w1_base = v[j1].data() + f1 * align0;
      real_t v1 = iter_i->feat_val;
      for (SparseRow::const_iterator iter_j = iter_i+1;
          iter_j != row->end(); ++iter_j) {
        index_t j2 = iter_j->feat_id;
        index_t f2 = iter_j->field_id;
        real_t v2 = iter_j->feat_val;
        real_t* w2_base = v[j2].data() + f2 * align0;
        __m128 XMMv = _mm_set1_ps(v1*v2);
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
          XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
          XMMwg2 = _mm_add_ps(XMMwg2, _mm_mul_ps(XMMg2, XMMg2));
          _mm_store_ps(wg1, XMMwg1);
          _mm_store_ps(wg2, XMMwg2);
        }
      }
    }
  }
}

// Calculate gradient and update current model using ftrl
void DistFFMScore::calc_grad_ftrl(const DMatrix* matrix,
    Model& model,
    std::map<index_t, real_t>& weight,
    std::map<index_t, std::vector<real_t>>& v,
    real_t* sum,
    std::map<index_t, real_t>& w_g,
    std::map<index_t, std::vector<real_t>>& v_g,
    index_t start_idx,
    index_t end_idx
    ) {
  for (index_t i = start_idx; i < end_idx; ++i) {
    SparseRow* row = matrix->row[i];
    real_t pred = CalcScore(row, model, &w, &v);
    real_t y = matrix->Y[i] > 0 ? 1.0 : -1.0;
    (*sum) += log1p(exp(-y*pred));
    real_t pg = -y / (1.0 + (1.0 / exp(-y * pred)));
    /*********************************************************
     *  linear term and bias term                            *
     *********************************************************/  
    for (SparseRow::const_iterator iter = row->begin();
        iter != row->end(); ++iter) {
      real_t &wl = weight[iter->feat_id];
      real_t g = lambda_2_*wl+pg*iter->feat_val; 
      v_g[iter->feat_id] += g;
    }
    /*********************************************************
     *  latent factor                                        *
     *********************************************************/
    index_t align0 = model.get_aligned_k();  // latent factor dim
    index_t align1 = model.GetNumField() * align0;  // all latent factor dim for one feature;
    index_t align = kAlign;
    //w = model.GetParameter_v();
    __m128 XMMpg = _mm_set1_ps(pg);
    __m128 XMMalpha = _mm_set1_ps(alpha_);
    __m128 XMML2 = _mm_set1_ps(lambda_2_);
    for (SparseRow::const_iterator iter_i = row->begin();
        iter_i != row->end(); ++iter_i) {
      index_t j1 = iter_i->feat_id;
      index_t f1 = iter_i->field_id;
      real_t v1 = iter_i->feat_val;
      real_t* w1_base = v[j1].data() + f1*align0;
      for (SparseRow::const_iterator iter_j = iter_i+1;
          iter_j != row->end(); ++iter_j) {
        index_t j2 = iter_j->feat_id;
        index_t f2 = iter_j->field_id;
        real_t v2 = iter_j->feat_val;
        real_t* w2_base = v[j2].data() + f2*align0;
        __m128 XMMv = _mm_set1_ps(v1*v2);
        __m128 XMMpgv = _mm_mul_ps(XMMv, XMMpg);
        for (index_t d = 0; d < align0; d += align) {
          real_t *w1 = w1_base + d;
          real_t *w2 = w2_base + d;
          __m128 XMMw1 = _mm_load_ps(w1);
          __m128 XMMw2 = _mm_load_ps(w2);
          __m128 XMMg1 = _mm_add_ps(
              _mm_mul_ps(XMML2, XMMw1),
              _mm_mul_ps(XMMpgv, XMMw2));
          __m128 XMMg2 = _mm_add_ps(
              _mm_mul_ps(XMML2, XMMw2),
              _mm_mul_ps(XMMpgv, XMMw1));
          XMMwg1 = _mm_add_ps(XMMwg1,
              _mm_mul_ps(XMMg1, XMMg1));
          XMMwg2 = _mm_add_ps(XMMwg2,
              _mm_mul_ps(XMMg2, XMMg2));
          _mm_store_ps(v_g[j1].data() + d, XMMwg1);
          _mm_store_ps(v_g[j2].data() + d, XMMwg2);
        }
      }
    }
  }
}

} // namespace xLearn
