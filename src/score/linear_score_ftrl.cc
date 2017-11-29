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

#include "src/score/linear_score_ftrl.h"
#include "src/base/math.h"

namespace xLearn {

// y = wTx (incluing bias term)
real_t LinearScoreFtrl::CalcScore(const SparseRow* row,
                              Model& model,
                              real_t norm) {
  real_t* w = model.GetParameter_w();
  real_t score = 0.0;
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t idx = iter->feat_id * 3;
    score += w[idx] * iter->feat_val;
  }
  // bias
  score += model.GetParameter_b()[0];
  return score;
}

// Calculate gradient and update current model
void LinearScoreFtrl::CalcGrad(const SparseRow* row,
                           Model& model,
                           real_t pg,
                           real_t norm) {
  real_t alpha = 0.01;
  real_t beta = 1.0;
  real_t lambda1 = 0.1;
  real_t lambda2 = 1.0;
  real_t* w = model.GetParameter_w();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    real_t gradient = pg * iter->feat_val;
    index_t idx_w = iter->feat_id * 3; // index of w_i
    index_t idx_n = idx_w + 1; // index of cumulate for gradient * gradient
    index_t idx_z = idx_w + 2; // index of cumulater for gradient
    real_t old_n = w[idx_n]; // old n_i
    w[idx_n] += (gradient * gradient); // new n_i
    real_t sigma = 1.0f * (std::sqrt(w[idx_n]) - sqrt(old_n))
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
  wbz += g;
  if (std::abs(wbz) <= lambda1) {
    wb = 0.0f;
  } else {
    real_t smooth_lr = 1.0f
                       / (lambda2 + (beta + std::sqrt(wbn)) / alpha);
    if (wbz < 0.0) {
      wbz += lambda1;
    } else if(wbz > 0.0) {
      wbz -= lambda1;
    }
    wb = -1.0f * smooth_lr * wbz;
  }
}

} // namespace xLearn
