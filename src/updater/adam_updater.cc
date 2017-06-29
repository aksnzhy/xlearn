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

This file is the implementation of Adam updater.
*/

#include "src/updater/adam_updater.h"

namespace xLearn {

// This function needs to be invoked before using this class.
void Adam::Initialize(const HyperParam& hyper_param) {
  CHECK_GT(hyper_param.learning_rate, 0);
  CHECK_GT(hyper_param.regu_lambda_1, 0);
  CHECK_GT(hyper_param.regu_lambda_2, 0);
  CHECK_GE(hyper_param.decay_rate, 0);
  CHECK_GE(hyper_param.second_decay_rate, 0);
  CHECK_GT(hyper_param.batch_size, 0);
  learning_rate_ = hyper_param.learning_rate;
  regu_lambda_1_ = hyper_param.regu_lambda_1;
  regu_lambda_2_ = hyper_param.regu_lambda_2;
  regu_type_ = hyper_param.regu_type;
  beta1_ = hyper_param.decay_rate;
  beta2_ = hyper_param.second_decay_rate;
  count_num_ = hyper_param.batch_size;
  // Allocating memory for two vectors
  try {
    m_.resize(hyper_param.num_param, 0.0);
    v_.resize(hyper_param.num_param, 0.0);
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for current    \
                   model parameters. Parameter size: "
               << hyper_param.num_param;
  }
}

// Adam updater
// [ m = beta1 * m + (1-beta1) * dx ]
// [ v = beta2 * v + (1-beta2) * (dx ^ 2) ]
// [ w -= learning_rate * m / (sqrt(v) + 1e-7) ]
void Adam::Update(const index_t id,
                  const real_t grad,
                  std::vector<real_t>& param) {
  // Do not check anything here
  static int n = 1;
  static int tmp_n = 0;
  m_[id] = (1-beta1_) * grad + beta1_ * m_[id];
  v_[id] = (1-beta2_) * grad * grad + beta2_ * v_[id];
  real_t mb = m_[id] / (1-fastpow(beta1_, n));
  real_t vb = v_[id] / (1-fastpow(beta2_, n));
  param[id] -= learning_rate_ * mb * InvSqrt(vb);
  tmp_n++;
  if (tmp_n == count_num_) { n++; tmp_n = 0; }
}

// Update a continous space of model parameters by
// using sse/avx to speed up.
void Adam::BatchUpdate(const std::vector<real_t>& value,
                       const index_t start_id,
                       std::vector<real_t>& param) {
  static int n = 1;
  static int tmp_n = 0;
  CHECK_EQ(value.empty(), false);
  // Ensuring for sse/avx
  CHECK_EQ(value.size() % _MMX_INCREMENT, 0);
  __MX _learning_rate = _MMX_SET1_PS(learning_rate_);
  __MX _beta_1 = _MMX_SET1_PS(beta1_);
  __MX _1_minus_beta_1 = _MMX_SET1_PS(1-beta1_);
  __MX _1_minus_pow_beta_1 = _MMX_SET1_PS(1-fastpow(beta1_, n));
  __MX _bata_2 = _MMX_SET1_PS(beta2_);
  __MX _1_minus_beta_2 = _MMX_SET1_PS(1-beta2_);
  __MX _1_minus_pow_beta_2 = _MMX_SET1_PS(1-fastpow(beta2_, n));
  __MX _small_num = _MMX_SET1_PS(kVerySmallNumber);
  for (size_t i = 0; i < value.size(); i += _MMX_INCREMENT) {
    index_t id = start_id + i;
    __MX _grad = _MMX_LOAD_PS(value.data() + i);
    __MX _v = _MMX_LOAD_PS(v_.data() + id);
    __MX _m = _MMX_LOAD_PS(m_.data() + id);
    __MX _w = _MMX_LOAD_PS(param.data() + id);
    // [ m = beta1 * m + (1-beta) * dx ]
    // [ v = beta2 * v + (1-beta) * (dx^2) ]
    // [ w -= learning_rate * m / (sqrt(v) + 1e-7) ]
    _m = _MMX_ADD_PS(_MMX_MUL_PS(_1_minus_beta_1, _grad),
                     _MMX_MUL_PS(_beta_1, _m));
    _MMX_STORE_PS(m_.data() + id, _m);
    _v = _MMX_ADD_PS(_MMX_MUL_PS(_1_minus_beta_2, _grad),
                     _MMX_MUL_PS(_bata_2, _v));
    _MMX_STORE_PS(v_.data() + id, _v);
    __MX _mb = _MMX_DIV_PS(_m, _1_minus_pow_beta_1);
    __MX _vb = _MMX_DIV_PS(_v, _1_minus_pow_beta_2);
    _w = _MMX_SUB_PS(_w, _MMX_MUL_PS(_learning_rate,
                         _MMX_MUL_PS(_mb,
                         _MMX_RSQRT_PS(_MMX_ADD_PS(_vb,
                                       _small_num)))));
    _MMX_STORE_PS(param.data() + id, _w);
  }
  tmp_n++;
  if (tmp_n == count_num_) { n++; tmp_n = 0; }
}

} // namespace xLearn
