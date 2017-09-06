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
Author: Yuze Liao and Chao Ma (mctt90@gmail.com)

This file is the implementation of Updater.
*/

/* for class register */
#include "src/base/math.h"
#include "src/updater/updater.h"
#include "src/updater/adam_updater.h"
#include "src/updater/adagrad_updater.h"
#include "src/updater/adadelta_updater.h"
#include "src/updater/momentum_updater.h"
#include "src/updater/nesterov_updater.h"
#include "src/updater/rmsprop_updater.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_updater_registry, Updater);
REGISTER_UPDATER("sgd", Updater);
REGISTER_UPDATER("adam", Adam);
REGISTER_UPDATER("adagrad", AdaGrad);
REGISTER_UPDATER("adadelta", AdaDelta);
REGISTER_UPDATER("momentum", Momentum);
REGISTER_UPDATER("nesterov", Nesterov);
REGISTER_UPDATER("rmsprop", RMSProp);

// User need to invoke this function before updating.
void Updater::Initialize(const HyperParam& hyper_param) {
  CHECK_GT(hyper_param.learning_rate, 0);
  CHECK_GE(hyper_param.regu_lambda_, 0);
  learning_rate_ = hyper_param.learning_rate;
  regu_lambda_ = hyper_param.regu_lambda_;
  _lr = _MMX_SET1_PS(learning_rate_);
  _lambda = _MMX_SET1_PS(regu_lambda_);
}

// SGD updater: [w -= learning_rate * gradient]
void Updater::Update(const index_t id,
                     const real_t grad,
                     std::vector<real_t>& param) {
  // Do not check anything here
  param[id] -= (learning_rate_*grad +
                regu_lambda_1_*param[id]);
}

// Update a continuous space of model parameters by
// using SSE/AVX to speed up.
void Updater::BatchUpdate(const std::vector<real_t>& value,
                          const index_t start_id,
                          std::vector<real_t>& param) {
  // Do not check anything here
  for (size_t i = 0; i < value.size(); i += _MMX_INCREMENT) {
    index_t id = start_id + i;
    __MX _grad = _MMX_LOAD_PS(value.data() + i);
    __MX _w = _MMX_LOAD_PS(param.data() + id);
    // w -= learning_rate * grad
    _MMX_STORE_PS(param.data() + id,
                  _MMX_SUB_PS(_w,
                  _MMX_MUL_PS(_lr, _grad)));
  }
}

// L1 regularize
void Updater::Regularize_L1(Model* model) {
  CHECK_NOTNULL(model);
  std::vector<real_t>* w = model->GetParameter();
  for (size_t i = 0; i < w->size(); ++i) {
    if ((*w)[i] > 0) {
      (*w)[i] -= regu_lambda_1_;
    } else {
      (*w)[i] += regu_lambda_1_;
    }
  }
}

// L2 regularize. Using AVX/SSE to speed up
void Updater::Regularize_L2(Model* model) {
  CHECK_NOTNULL(model);
  std::vector<real_t>* w = model->GetParameter();
  CHECK_EQ(w->size() % _MMX_INCREMENT, 0);
  __MX _regu_lambda = _MMX_SET1_PS(regu_lambda_1_);
  // w -= regu_lambda_ * w
  for (size_t i = 0; i < w->size(); i += _MMX_INCREMENT) {
    __MX _w = _MMX_LOAD_PS(w->data() + i);
    __MX _delta_w = _MMX_MUL_PS(_regu_lambda, _w);
    _MMX_STORE_PS(w->data() + i,
                  _MMX_SUB_PS(_w, _delta_w));
  }
}

// ElasticNet (L1_L2) regularize
void Updater::Regularize_ElasticNet(Model* model) {
  CHECK_NOTNULL(model);
  std::vector<real_t>* w = model->GetParameter();
  CHECK_EQ(w->size() % _MMX_INCREMENT, 0);
  // l1
  for (size_t i = 0; i < w->size(); ++i) {
    if ((*w)[i] > 0) {
      (*w)[i] -= regu_lambda_1_;
    } else {
      (*w)[i] += regu_lambda_1_;
    }
  }
  // l2
  __MX _regu_lambda = _MMX_SET1_PS(regu_lambda_2_);
  for (size_t i = 0; i < w->size(); i += _MMX_INCREMENT) {
    __MX _w = _MMX_LOAD_PS(w->data() + i);
    __MX _delta_w = _MMX_MUL_PS(_regu_lambda, _w);
    _MMX_STORE_PS(w->data() + i,
                  _MMX_SUB_PS(_w, _delta_w));
  }
}

} // namespace xLearn
