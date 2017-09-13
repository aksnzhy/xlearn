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
#include "src/updater/momentum_updater.h"
#include "src/updater/rmsprop_updater.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_updater_registry, Updater);
REGISTER_UPDATER("sgd", Updater);
REGISTER_UPDATER("adagrad", AdaGrad);
REGISTER_UPDATER("adadelta", AdaDelta);
REGISTER_UPDATER("momentum", Momentum);
REGISTER_UPDATER("rmsprop", RMSProp);

// User need to invoke this function before updating.
void Updater::Initialize(real_t learning_rate,
                         real_t regu_lambda,
                         real_t decay_rate_1 = 0,
                         real_t decay_rate_2 = 0,
                         index_t num_param = 0) {
  CHECK_GT(learning_rate, 0);
  // regu_lambda == 0 means that we will not use regularizer
  CHECK_GE(regu_lambda, 0);
  learning_rate_ = learning_rate;
  regu_lambda_ = regu_lambda;
  _lr = _MMX_SET1_PS(learning_rate_);
  _lambda = _MMX_SET1_PS(regu_lambda_);
}

// SGD updater: [w -= learning_rate * gradient]
void Updater::Update(const index_t id,
                     const real_t grad,
                     std::vector<real_t>& param) {
  // Do not check anything here
  param[id] -= (learning_rate_*grad +      // grad
                regu_lambda_*param[id]);   // regular
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
                    _MMX_ADD_PS(
                      _MMX_MUL_PS(_lr, _grad),
                      _MMX_MUL_PS(_lambda, _w)
                    )
                   )
                 );
  }
}

void Updater::BatchUpdate(__MX w_k, __MX grad,  real_t* w) {
  grad = _MMX_ADD_PS(_MMX_MUL_PS(_lr, grad), _MMX_MUL_PS(_lambda, w_k));
  _MMX_STORE_PS(w, _MMX_SUB_PS(w_k, grad));
}

} // namespace xLearn
