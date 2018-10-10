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
  index_t num_feat = model.GetNumFeature();
  real_t score = 0.0;
  index_t auxiliary_size = model.GetAuxiliarySize();
  // linear term
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t feat_id = iter->feat_id;
    // To avoid unseen feature in Prediction
    if (feat_id >= num_feat) continue;
    index_t idx = feat_id * auxiliary_size;
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
  else {
    LOG(FATAL) << "Unknow optimization method: " << opt_type_;
  }
}

// Calculate gradient and update current model using sgd
// TODO(aksnzhy): solve unseen feature
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
// TODO(aksnzhy): solve unseen feature
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
// TODO(aksnzhy): solve unseen feature
void LinearScore::calc_grad_ftrl(const SparseRow* row,
                                 Model& model,
                                 real_t pg,
                                 real_t norm) {
  // linear term
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
}

} // namespace xLearn
