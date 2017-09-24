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

#include "src/base/math.h" // for sqrt()

namespace xLearn {

// This function needs to be invoked before update.
void AdaDelta::Initialize(real_t learning_rate,
                      real_t regu_lambda,
                      real_t decay_rate,
                      index_t param_num_w) {
  CHECK_GT(learning_rate, 0);
  // regu_lambda == 0 means that we will not use regularizer
  CHECK_GE(regu_lambda, 0);
  CHECK_GT(decay_rate, 0);
  learning_rate_ = learning_rate;
  regu_lambda_ = regu_lambda;
  decay_rate_ = decay_rate;
  _lr = _MMX_SET1_PS(learning_rate_);
  _lambda = _MMX_SET1_PS(regu_lambda_);
  _decay_rate = _MMX_SET1_PS(decay_rate_);
  // Allocating memory for cache vector
  try {
    #ifdef __AVX__
      posix_memalign((void**)&cache_, 32,
         param_num_w * sizeof(real_t));
    #else // SSE
      posix_memalign((void**)&cache_, 16,
         param_num_w * sizeof(real_t));
    #endif
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for current    \
                   model parameters. Parameter size: "
               << param_num_w;
  }
  for (index_t i = 0; i < param_num_w; ++i) {
    cache_[i] = 0.0;
  }
}

// AdaDelta updater
// [ cache = (1-decay_rate) * (grad^2) + decay_rate * cache ]
// [ w -= learning_rate * grad  / sqrt(cache) ]
void AdaDelta::Update(const index_t idx,
                      const real_t grad,
                      real_t* w) {
  // Do not check anything here
  real_t grad_w = grad + regu_lambda_ * w[idx];
  cache_[idx] = (1-decay_rate_) * (grad_w * grad_w)
                 + decay_rate_ * cache_[idx];
  w[idx] -= learning_rate_ * grad * InvSqrt(cache_[idx]);
}

// Update a continous space of model parameters by
// using sse/avx to speed up.
void AdaDelta::BatchUpdate(__MX _w,
                           __MX _grad,
                           index_t idx,
                           real_t* w) {
  // Do not check anything here
  static __MX _1_minus_decay_rate = _MMX_SET1_PS(1-decay_rate_);
  _grad = _MMX_ADD_PS(_grad,
          _MMX_MUL_PS(_lambda, _w));
  __MX _cache = _MMX_LOAD_PS(cache_ + idx);
  // [ cache = (1-decay_rate) * (grad^2) + decay_rate * cache ]
  _cache = _MMX_ADD_PS(_MMX_MUL_PS(_1_minus_decay_rate,
           _MMX_MUL_PS(_grad, _grad)),
           _MMX_MUL_PS(_decay_rate, _cache));
  _MMX_STORE_PS(cache_ + idx, _cache);
  // [ w -= learning_rate * grad  / sqrt(cache) ]
  _MMX_STORE_PS(w + idx,
    _MMX_SUB_PS(_w,
        _MMX_MUL_PS(_lr,
          _MMX_MUL_PS(_grad,
            _MMX_RSQRT_PS(_cache)))));
}

}// namespace xLearn
