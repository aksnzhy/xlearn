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

#include "src/score/fm_score.h"
#include "src/base/math.h"

namespace xLearn {

// y = wTx + sum[(V_i*V_j)(x_i * x_j)]
real_t FMScore::CalcScore(const SparseRow* row,
                          Model& model) {
  real_t score = 0.0;
  /*********************************************************
   *  linear term                                          *
   *********************************************************/
  real_t *w = model.GetParameter_w();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t idx = iter->feat_id;
    score += w[idx] * iter->feat_val;
  }
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  real_t tmp = 0.0;
  static index_t num_factor = model.GetNumK();
  w = model.GetParameter_w() + model.GetNumFeature();
  for (index_t k = 0; k < num_factor; ++k) {
    real_t square_sum = 0.0;
    real_t sum_square = 0.0;
    for (SparseRow::const_iterator iter = row->begin();
         iter != row->end(); ++iter) {
      real_t x = iter->feat_val;
      index_t pos = iter->feat_id * num_factor + k;
      real_t v = w[pos];
      real_t x_v = x*v;
      square_sum += x_v;
      sum_square += x_v*x_v;
    }
    square_sum *= square_sum;
    tmp += (square_sum - sum_square);
  }
  score += (0.5 * tmp);
  return score;
}

// Calculate gradient and update current
// model parameters
void FMScore::CalcGrad(const SparseRow* row,
                       Model& model,
                       real_t pg) {
 /*********************************************************
  *  linear term                                          *
  *********************************************************/
  real_t *w = model.GetParameter_w();
  real_t* cache = model.GetParameter_cache();
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
  static index_t num_factor = model.GetNumK();
  w = model.GetParameter_w() + model.GetNumFeature();
  cache = model.GetParameter_cache() + model.GetNumFeature();
  for (size_t k = 0; k < num_factor; ++k) {
    real_t v_mul_x = 0.0;
    for (SparseRow::const_iterator iter = row->begin();
         iter != row->end(); ++iter) {
      index_t pos = iter->feat_id * num_factor + k;
      real_t v = w[pos];
      real_t x = iter->feat_val;
      v_mul_x += (x*v);
    }
    for (SparseRow::const_iterator iter = row->begin();
         iter != row->end(); ++iter) {
      index_t pos = iter->feat_id * num_factor + k;
      real_t v = w[pos];
      real_t x = iter->feat_val;
      real_t gradient = x*(v_mul_x-v*x) * pg;
      gradient += regu_lambda_ * w[pos];
      cache[pos] += (gradient * gradient);
      w[pos] -= (learning_rate_ * gradient *
                 InvSqrt((cache)[pos]));
    }
  }
}

} // namespace xLearn
