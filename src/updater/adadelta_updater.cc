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

This file is the implementation of AdaDelta updater.
*/

#include "src/updater/adadelta_updater.h"

#include <cmath> // for sqrt()

namespace xLearn {

// This function needs to be invoked before update.
void AdaDelta::Initialize(const HyperParam& hyper_param) {
  CHECK_GT(hyper_param.learning_rate, 0);
  CHECK_GT(hyper_param.regu_lambda_1, 0);
  CHECK_GT(hyper_param.regu_lambda_2, 0);
  CHECK_GE(hyper_param.decay_rate, 0);
  learning_rate_ = hyper_param.learning_rate;
  regu_lambda_1_ = hyper_param.regu_lambda_1;
  regu_lambda_2_ = hyper_param.regu_lambda_2;
  regu_type_ = hyper_param.regu_type;
  decay_rate_ = hyper_param.decay_rate;
  _lr = _MMX_SET1_PS(learning_rate_);
  // Allicating memory for cache vector
  try {
    cache_.resize(hyper_param.num_param, 0.0);
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for current    \
                   model parameters. Parameter size: "
               << hyper_param.num_param;
  }
}

// AdaDelta updater
// [ cache = (1-decay_rate) * (grad^2) + decay_rate * cache ]
// [ w -= learning_rate * grad  / sqrt(cache) ]
void AdaDelta::Update(const index_t id,
                      const real_t grad,
                      std::vector<real_t>& param) {
  // Do not check anything here
  cache_[id] = (1-decay_rate_) * (grad * grad)
               + decay_rate_ * cache_[id];
  param[id] -= learning_rate_ * grad * InvSqrt(cache_[id]);
}

// Update a continous space of model parameters by
// using sse/avx to speed up.
void AdaDelta::BatchUpdate(const std::vector<real_t>& value,
                           const index_t start_id,
                           std::vector<real_t>& param) {
  // Do not check anything here
  __MX _small_num = _MMX_SET1_PS(kVerySmallNumber);
  __MX _1_minus_decay_rate = _MMX_SET1_PS(1-decay_rate_);
  __MX _decay_rate = _MMX_SET1_PS(decay_rate_);
  for (size_t i = 0; i < value.size(); i += _MMX_INCREMENT) {
    index_t id = start_id + i;
    __MX _grad = _MMX_LOAD_PS(value.data() + i);
    __MX _w = _MMX_LOAD_PS(param.data() + id);
    __MX _cache = _MMX_LOAD_PS(cache_.data() + id);
    // [ cache = (1-decay_rate) * (grad^2) + decay_rate * cache ]
    // [ w -= learning_rate * grad  / sqrt(cache) ]
    _cache = _MMX_ADD_PS(_MMX_MUL_PS(
             _MMX_MUL_PS(_1_minus_decay_rate, _grad), _grad),
             _MMX_MUL_PS(_decay_rate, _cache));
    _MMX_STORE_PS(cache_.data() + id, _cache);
    _MMX_STORE_PS(param.data() + id,
                  _MMX_SUB_PS(_w,
                  _MMX_MUL_PS(_lr,
                  _MMX_MUL_PS(_grad,
                  _MMX_RSQRT_PS(
                  _MMX_ADD_PS(_cache, _small_num))))));
  }
}

}// namespace xLearn
