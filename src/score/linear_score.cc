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
This file is the implementation of LinearScore class.
*/

#include "src/score/linear_score.h"
#include "src/base/math.h"

namespace xLearn {

// y = wTx (bias is added in w and x automitically)
real_t LinearScore::CalcScore(const SparseRow* row,
                              Model& model,
                              real_t norm) {
  real_t* w = model.GetParameter_w();
  real_t score = 0.0;
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t idx = iter->feat_id * 2;
    score += w[idx] * iter->feat_val;
  }
  // bias
  score += model.GetParameter_b()[0];
  return score;
}

// Calculate gradient and update current model
void LinearScore::CalcGrad(const SparseRow* row,
                           Model& model,
                           real_t pg,
                           real_t norm) {
  real_t* w = model.GetParameter_w();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    real_t gradient = pg * iter->feat_val;
    index_t idx_g = iter->feat_id * 2;
    index_t idx_c = idx_g + 1;
    gradient += regu_lambda_ * w[idx_g];
    w[idx_c] += (gradient * gradient);
    w[idx_g] -= (learning_rate_ * gradient *
                 InvSqrt(w[idx_c]));
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t &wbg = w[1];
  real_t g = pg;
  wbg += g*g;
  wb -= learning_rate_ * g * InvSqrt(wbg);
}

} // namespace xLearn
