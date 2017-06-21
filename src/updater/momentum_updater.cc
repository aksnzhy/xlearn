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

#include "src/update/momentum_updater.h"

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
    v_.resize(hyper_param.num_param);
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for current    \
                   model parameters. Parameter size: "
               << hyper_param.num_param;
  }
}

// Momentum updater:
// [ v = rho * v + dx ]
// [ w -= learning_rate * v]
void Momentum::Update(const real_t grad, real_t* param) {
  // Do not check anything here
  std::vector<real_t>* w = model->GetParameter();
  std::vector<real_t>* v = model->GetParamCache();
  real_t tmp = RegularTerm((*w)[key]) + grad;
  (*v)[key] = mu_ * (*v)[key] - learning_rate_ * tmp;
  (*w)[key] += (*v)[key];
}

// Update model parameter in a mini-batch GD.
// Using SSE to speed up.
void Momentum::BatchUpdate(Gradient* grad, Model* model) {
  // g /= row_len
  size_t end = model->GetLength();
  grad->Div(grad->GetMiniBatchSize());
  std::vector<real_t>* w = model->GetParameter();
  std::vector<real_t>* v = model->GetParamCache();
  std::vector<real_t>* value = grad->GetDenseVector();
  __MX _learning_rate = _MMX_SET1_PS(learning_rate_);
  __MX _regu_lambda = _MMX_SET1_PS(regu_lambda_);
  __MX _mu = _MMX_SET1_PS(mu_);
  for (size_t start_key = 0; start_key < end; start_key += _MMX_INCREMENT) {
     __MX _regular_term;
    GetRegularTerm(_regular_term, regu_type_);
    __MX _v = _MMX_LOAD_PS(&(*v)[start_key]);
    __MX _w = _MMX_LOAD_PS(&(*w)[start_key]);
    __MX _tmp = _MMX_ADD_PS(_MMX_MUL_PS(_regu_lambda, _regular_term),
                             _MMX_LOAD_PS(&(*value)[start_key]));
    _v = _MMX_SUB_PS(_MMX_MUL_PS(_mu, _v), _MMX_MUL_PS(_learning_rate, _tmp));
    _MMX_STORE_PS(&(*w)[start_key], _MMX_ADD_PS(_w, _v));
    _MMX_STORE_PS(&(*v)[start_key], _v);
  }
}

// Update a continuous model parameter.
// Using SSE to speed up.
void Momentum::SeqUpdate(std::vector<real_t>& value,
                                index_t start_key,
                                Model* model) {
  // Do not check anything here
  index_t end = value.size();
  std::vector<real_t>* w = model->GetParameter();
  std::vector<real_t>* v = model->GetParamCache();
  __MX _learning_rate = _MMX_SET1_PS(learning_rate_);
  __MX _regu_lambda = _MMX_SET1_PS(regu_lambda_);
  __MX _mu = _MMX_SET1_PS(mu_);
  for (index_t i = 0; i < end; i += _MMX_INCREMENT) {
    __MX _regular_term;
    GetRegularTerm(_regular_term, regu_type_);
    __MX _v = _MMX_LOAD_PS(&(*v)[start_key]);
    __MX _w = _MMX_LOAD_PS(&(*w)[start_key]);
    __MX _tmp = _MMX_ADD_PS(_MMX_MUL_PS(_regu_lambda, _regular_term), _MMX_LOAD_PS(&value[i]));
    _v = _MMX_SUB_PS(_MMX_MUL_PS(_mu, _v), _MMX_MUL_PS(_learning_rate, _tmp));
    _MMX_STORE_PS(&(*w)[start_key], _MMX_ADD_PS(_w, _v));
    _MMX_STORE_PS(&(*v)[start_key], _v);
    start_key += _MMX_INCREMENT;
  }
}

}// namespace xLearn
