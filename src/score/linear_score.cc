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

// y = wTx (incluing bias term)
real_t LinearScore::CalcScore(const SparseRow* row,
                              Model& model,
                              real_t norm) {
  real_t* w = model.GetParameter_w();
  real_t score = 0.0;
  index_t auxiliary_size = model.GetAuxiliarySize();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t idx = iter->feat_id * auxiliary_size;
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
void LinearScore::calc_grad_sgd(const SparseRow* row,
                                Model& model,
                                real_t pg,
                                real_t norm) {
  // linear term
  real_t* w = model.GetParameter_w();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    real_t gradient = pg * iter->feat_val;
    index_t idx_g = iter->feat_id;
    gradient += regu_lambda_ * w[idx_g];
    w[idx_g] -= (learning_rate_ * gradient);
  }
  // bias
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t g = pg;
  wb -= learning_rate_ * g;
}

// Calculate gradient and update current model using adagrad
void LinearScore::calc_grad_adagrad(const SparseRow* row,
                                    Model& model,
                                    real_t pg,
                                    real_t norm) {
  // linear term
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

// Calculate gradient and update current model using ftrl
void LinearScore::calc_grad_ftrl(const SparseRow* row,
                                 Model& model,
                                 real_t pg,
                                 real_t norm) {
  real_t* w = model.GetParameter_w();
  for (SparseRow::const_iterator iter = row->begin();
      iter != row->end(); ++iter) {
    real_t gradient = pg * iter->feat_val;
    index_t idx_w = iter->feat_id * 3;
    index_t idx_n = idx_w + 1;
    index_t idx_z = idx_w + 2;
    real_t old_n = w[idx_n];
    w[idx_n] += (gradient * gradient);
    real_t sqrt_n = sqrt(w[idx_n]);
    real_t sigma = (sqrt_n - sqrt(old_n))
                   / alpha_;
    w[idx_z] += gradient - sigma * w[idx_w];
    if (std::abs(w[idx_z]) <= lambda_1_) {
      w[idx_w] = 0.0;
    } else {
      real_t smooth_lr = -1.0f
                         / (lambda_2_ + (beta_ + sqrt_n) / alpha_);
      if (w[idx_z] > 0.0) {
        w[idx_z] -= lambda_1_;
      }
      if (w[idx_z] < 0.0) {
        w[idx_z] += lambda_1_;
      }
      w[idx_w] = smooth_lr * w[idx_z];
    }
  }
  w = model.GetParameter_b();
  real_t &wb = w[0];
  real_t &wbn = w[1];
  real_t &wbz = w[2];
  real_t g = pg;
  real_t old_n = wbn;
  wbn += g*g;
  real_t sqrt_wbn = sqrt(wbn);
  real_t sigma_wbn = (sqrt_wbn - sqrt(old_n)) / alpha_;
  wbz += g - sigma_wbn * wb;
  if (std::abs(wbz) <= lambda_1_) {
    wb = 0.0f;
  } else {
    real_t smooth_lr = -1.0f
                       / (lambda_2_ + (beta_ + sqrt_wbn) / alpha_);
    if (wbz > 0.0) {
      wbz -= lambda_1_;
    }
    if (wbz < 0.0) {
      wbz += lambda_1_;
    }
    wb = smooth_lr * wbz;
  }
}

} // namespace xLearn
