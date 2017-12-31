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

#include "src/distributed/dist_fm_score.h"
#include "src/base/math.h"

namespace xLearn {

// y = sum( (V_i*V_j)(x_i * x_j) )
// Using SSE to accelerate vector operation.
real_t DistFMScore::CalcScore(const SparseRow* row,
      Model& model,
      std::map<index_t, real_t>* weight,
      std::map<index_t, std::vector<real_t>>* v,
      real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = sqrt(norm);
  real_t t = 0;
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    t += (iter->feat_val * ((*weight)[iter->feat_id]) *
          sqrt_norm);
  }
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  index_t aligned_k = model.get_aligned_k();
  std::vector<real_t> sv(aligned_k, 0);
  real_t* s = sv.data();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t j1 = iter->feat_id;
    real_t v1 = iter->feat_val;
    real_t *w = (*v)[j1].data();
    __m128 XMMv = _mm_set1_ps(v1*norm);
    real_t* s = (*v)[j1].data();
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
    real_t* w = (*v)[j1].data();
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
void DistFMScore::DistCalcGrad(const DMatrix* matrix,
    Model& model,
    std::map<index_t, real_t>& w,
    std::map<index_t, std::vector<real_t>>& v,
    real_t* sum,
    std::map<index_t, real_t>& w_g,
    std::map<index_t, std::vector<real_t>>& v_g,
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
void DistFMScore::calc_grad_sgd(const DMatrix* matrix,
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
    real_t pred = CalcScore(row, model, &weight, &v);
    real_t y = matrix->Y[i] > 0 ? 1.0 : -1.0;
    (*sum) += log1p(exp(-y*pred));
    real_t pg = -y / (1.0 + (1.0 / exp(-y * pred)));
    /*********************************************************
     *  linear term and bias term                            *
     *********************************************************/  
    for (SparseRow::const_iterator iter = row->begin();
         iter != row->end(); ++iter) {
      real_t g = pg*iter->feat_val;
      w_g[iter->feat_id] += g;
    }
    /*********************************************************
     *  latent factor                                        *
     *********************************************************/
    index_t aligned_k = model.get_aligned_k();
    __m128 XMMpg = _mm_set1_ps(pg);
    std::vector<real_t> sv(aligned_k, 0);
    real_t* s = sv.data();
    for (SparseRow::const_iterator iter = row->begin();
        iter != row->end(); ++iter) {
      index_t j1 = iter->feat_id;
      real_t v1 = iter->feat_val;
      real_t* w = v[j1].data();
      __m128 XMMv = _mm_set1_ps(v1);
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
      real_t* w = v[j1].data();
      __m128 XMMv = _mm_set1_ps(v1);
      __m128 XMMpgv = _mm_mul_ps(XMMpg, XMMv);
      real_t* v_gradient = v_g[j1].data();
      for(index_t d = 0; d < aligned_k; d += kAlign) {
        __m128 XMMs = _mm_load_ps(s+d);
        __m128 XMMw = _mm_load_ps(w+d);
        __m128 XMMg = _mm_mul_ps(XMMpgv, _mm_sub_ps(XMMs,
                _mm_mul_ps(XMMw, XMMv)));
        _mm_store_ps(v_gradient+d, XMMg);
      }
    }
  }
}

// Calculate gradient and update current model using adagrad
void DistFMScore::calc_grad_adagrad(const DMatrix* matrix,
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
    real_t pred = CalcScore(row, model, &weight, &v);
    real_t y = matrix->Y[i] > 0 ? 1.0 : -1.0;
    (*sum) += log1p(exp(-y*pred));
    real_t pg = -y / (1.0 + (1.0 / exp(-y * pred)));
    /*********************************************************
     *  linear term and bias term                            *
     *********************************************************/
    for (SparseRow::const_iterator iter = row->begin();
        iter != row->end(); ++iter) {
      real_t &wl = weight[iter->feat_id];
      real_t g = regu_lambda_*wl+pg*iter->feat_val;
      w_g[iter->feat_id] += g;
    }
    /*********************************************************
     *  latent factor                                        *
     *********************************************************/
    index_t aligned_k = model.get_aligned_k();
    __m128 XMMpg = _mm_set1_ps(pg);
    std::vector<real_t> sv(aligned_k, 0);
    real_t* s = sv.data();
    for (SparseRow::const_iterator iter = row->begin();
        iter != row->end(); ++iter) {
      index_t j1 = iter->feat_id;
      real_t v1 = iter->feat_val;
      real_t *w = v[j1].data();
      __m128 XMMv = _mm_set1_ps(v1);
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
      real_t* w = v[j1].data();
      __m128 XMMv = _mm_set1_ps(v1);
      __m128 XMMpgv = _mm_mul_ps(XMMpg, XMMv);
      real_t* v_gradient = v_g[j1].data();
      for(index_t d = 0; d < aligned_k; d += kAlign) {
        __m128 XMMw = _mm_load_ps(w+d);
        __m128 XMMs = _mm_load_ps(s+d);
        __m128 XMMg = _mm_mul_ps(XMMpgv, _mm_sub_ps(XMMs,
                _mm_mul_ps(XMMw, XMMv)));
        _mm_store_ps(v_gradient+d, XMMg);
      }
    }
  }
}

// Calculate gradient and update current model using ftrl
void DistFMScore::calc_grad_ftrl(const DMatrix* matrix,
    Model& model,
    std::map<index_t, real_t>& w,
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
      real_t g = pg*iter->feat_val;
      w_g[iter->feat_id] += g;
    }
    /*********************************************************
     *  latent factor                                        *
     *********************************************************/
    index_t aligned_k = model.get_aligned_k();
    __m128 XMMpg = _mm_set1_ps(pg);
    std::vector<real_t> sv(aligned_k, 0);
    real_t* s = sv.data();
    for (SparseRow::const_iterator iter = row->begin();
        iter != row->end(); ++iter) {
      index_t j1 = iter->feat_id;
      real_t v1 = iter->feat_val;
      real_t *w = v[j1].data();
      __m128 XMMv = _mm_set1_ps(v1);
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
      real_t *w_base = v[j1].data();
      __m128 XMMv = _mm_set1_ps(v1);
      __m128 XMMpgv = _mm_mul_ps(XMMpg, XMMv);
      real_t* v_gradient = v_g[j1].data();
      for (index_t d = 0; d < aligned_k; d += kAlign) {
        real_t* w = w_base + d;
        __m128 XMMs = _mm_load_ps(s+d);
        __m128 XMMw = _mm_load_ps(w);
        __m128 XMMg = _mm_mul_ps(XMMpgv,
              _mm_sub_ps(XMMs,
                _mm_mul_ps(XMMw, XMMv)));
        // Update w. SSE mat not faster.
        _mm_store_ps(v_gradient+d, XMMg);
      }
    }
  }
}

} // namespace xLearn
