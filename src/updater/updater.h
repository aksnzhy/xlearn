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

This file defines the Updater class that is responsible for updating
current model parameters.
*/

#ifndef XLEARN_UPDATER_UPDATER_H_
#define XLEARN_UPDATER_UPDATER_H_

#include <vector>

#include "src/base/common.h"
#include "src/base/class_register.h"
#include "src/data/model_parameters.h"
#include "src/data/hyper_parameters.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Updater class is responsible for updating current model parameters, and
// it can be implemented by different update functions such as naive SGD,
// Momentum, Adadelta, AdaGard, RMSprop, Nesterov, Adam, and so on.
// We use the naive SGD updater by default: [ w -= learning_rate * gradient ]
// Updater is used for the Loss class. For the linear and fm score, we use
// Update() method to update model parameters, while using BatchUpdate()
// to update ffm model for speedup.
// We use sparse regularizer in the Updater.
//------------------------------------------------------------------------------
class Updater {
 public:
  // Constructor and Destructor.
  Updater() {  }
  virtual ~Updater() {  }

  // This function needs to be invoked before using this class.
  virtual void Initialize(real_t learning_rate,
                          real_t regu_lambda,
                          real_t decay_rate_1,
                          real_t decay_rate_2,
                          index_t num_param);

  // Using naive SGD updater by default.
  virtual void Update(const index_t id,
                      const real_t grad,
                      std::vector<real_t>& param);

  // Update a continuous space of model parameters using SSE/AVX.
  virtual void BatchUpdate(const std::vector<real_t>& value,
                           const index_t start_id,
                           std::vector<real_t>& param);

 protected:
  real_t learning_rate_;
  real_t regu_lambda_;
  __MX _lr;
  __MX _lambda;

 private:
  DISALLOW_COPY_AND_ASSIGN(Updater);
};

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_DEFINE_REGISTRY(xLearn_updater_registry, Updater);

#define REGISTER_UPDATER(format_name, updater_name)               \
  CLASS_REGISTER_OBJECT_CREATOR(                                  \
      xLearn_updater_registry,                                    \
      Updater,                                                    \
      format_name,                                                \
      updater_name)

#define CREATE_UPDATER(format_name)                               \
  CLASS_REGISTER_CREATE_OBJECT(                                   \
      xLearn_updater_registry,                                    \
      format_name)

} // namespace xLearn

#endif // XLEARN_UPDATER_UPDATER_H_
