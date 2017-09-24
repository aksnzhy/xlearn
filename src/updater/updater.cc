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
#include "src/updater/adagrad_updater.h"
#include "src/updater/adadelta_updater.h"
//#include "src/updater/momentum_updater.h"
//#include "src/updater/rmsprop_updater.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_updater_registry, Updater);
REGISTER_UPDATER("sgd", Updater);
REGISTER_UPDATER("adagrad", AdaGrad);
REGISTER_UPDATER("adadelta", AdaDelta);
//REGISTER_UPDATER("momentum", Momentum);
//REGISTER_UPDATER("rmsprop", RMSProp);

// User need to invoke this function before updating
void Updater::Initialize(real_t learning_rate,
                      real_t regu_lambda,
                      real_t decay_rate,
                      index_t num_param_w) {
  CHECK_GT(learning_rate, 0);
  // regu_lambda == 0 means that do not use regularizer
  CHECK_GE(regu_lambda, 0);
  learning_rate_ = learning_rate;
  regu_lambda_ = regu_lambda;
  _lr = _MMX_SET1_PS(learning_rate_);
  _lambda = _MMX_SET1_PS(regu_lambda_);
}

// SGD updater: [w -= learning_rate * gradient]
void Updater::Update(const index_t idx,
                     const real_t grad,
                     real_t* w) {
  // Do not check anything here
  w[idx] -= (learning_rate_ *
            (grad + regu_lambda_ * w[idx]));
}

// Update a continuous space of model parameters by
// using SSE/AVX to speed up.
void Updater::BatchUpdate(__MX _w,
                          __MX _grad,
                          index_t idx,
                          real_t* w) {
  // grad = lr * grad + lambda * w
  _grad = _MMX_MUL_PS(_lr,
            _MMX_ADD_PS(_grad,
             _MMX_MUL_PS(_lambda, _w)));
  // w -= grad
  _MMX_STORE_PS(w + idx, _MMX_SUB_PS(_w, _grad));
}

} // namespace xLearn
