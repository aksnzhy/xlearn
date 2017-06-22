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

This file is the implementations of Momentum updater.
*/

#include "src/updater/momentum_updater.h"

namespace xLearn {

// This function need to be invoked before Updating.
void Momentum::Initialize(const HyperParam& hyper_param) {
  CHECK_GT(hyper_param.learning_rate, 0);
  CHECK_GT(hyper_param.regu_lambda_1, 0);
  CHECK_GT(hyper_param.regu_lambda_2, 0);
  CHECK_GT(hyper_param.decay_rate, 0);
  learning_rate_ = hyper_param.learning_rate;
  regu_lambda_1_ = hyper_param.regu_lambda_1;
  regu_lambda_2_ = hyper_param.regu_lambda_2;
  regu_type_ = hyper_param.regu_type;
  rho_ = hyper_param.decay_rate;
  // Allocating memory for the velocity vector
  try {
    v_.resize(hyper_param.num_param, 0.0);
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for current    \
                   model parameters. Parameter size: "
               << hyper_param.num_param;
  }
}

// Momentum updater:
// [ v = rho * v + dx ]
// [ w -= learning_rate * v]
void Momentum::Update(const index_t id,
                      const real_t grad,
                      std::vector<real_t>& param) {
  // Do not check anything here
  v_[id] = rho_ * v_[id] + grad;
  param[id] -= learning_rate_ * v_[id];
}

// Update a continuous space of model parameters by
// using sse/avx to speed up.
void Momentum::BatchUpdate(const std::vector<real_t>& value,
                           const index_t start_id,
                           std::vector<real_t>& param) {
  CHECK_EQ(value.empty(), false);
  // Ensuring for sse/avx
  CHECK_EQ(value.size() % _MMX_INCREMENT, 0);
  __MX _learning_rate = _MMX_SET1_PS(learning_rate_);
  __MX _rho = _MMX_SET1_PS(rho_);
  // [ v = rho * v + grad ]
  // [ w -= learning_rate * v ]
  for (size_t i  = 0; i < value.size(); i += _MMX_INCREMENT) {
    __MX _grad = _MMX_LOAD_PS(value.data() + i);
    __MX _v = _MMX_LOAD_PS(v_.data() + start_id + i);
    __MX _w = _MMX_LOAD_PS(param.data() + start_id + i);
    __MX _tmp_v = _MMX_ADD_PS(_MMX_MUL_PS(_rho, _v),
                              _grad);
    _MMX_STORE_PS(param.data() + start_id + i,
                 _MMX_SUB_PS(_w,
                           _MMX_MUL_PS(_learning_rate, _tmp_v)));
    _MMX_STORE_PS(v_.data() + start_id + i,
                 _tmp_v);
  }
}

}// namespace xLearn
