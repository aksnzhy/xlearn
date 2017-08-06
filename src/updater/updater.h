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
#include "src/base/math.h"
#include "src/base/class_register.h"
#include "src/data/model_parameters.h"
#include "src/data/hyper_parameters.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Updater class is responsible for updating current model parameters, and
// it can be implemented by different update functions such as naive SGD,
// Momentum, Adadelta, AdaGard, RMSprop, Adam, and so on.
// We use the naive SGD updater by default: [ w -= learning_rate * gradient ]
//------------------------------------------------------------------------------
class Updater {
 public:
  // Constructor and Destructor.
  Updater() {  }
  virtual ~Updater() {  }

  // This function needs to be invoked before using this class.
  virtual void Initialize(const HyperParam& hyper_param);

  // Using naive SGD updater by default.
  virtual void Update(const index_t id,
                      const real_t grad,
                      std::vector<real_t>& param);

  // Update a continuous space of model parameters using SSE/AVX.
  virtual void BatchUpdate(const std::vector<real_t>& value,
                           const index_t start_id,
                           std::vector<real_t>& param);

  // Penalize model complexity
  inline void Regularizer(Model* model) {
    if (regu_type_ == "l1") {
      return Regularize_L1(model);
    }
    else if (regu_type_ == "l2") {
      return Regularize_L2(model);
    }
    else if (regu_type_ == "elastic_net") {
      return Regularize_ElasticNet(model);
    }
    else return;
  }

 protected:
  // A set of Regularizer
  void Regularize_L1(Model* model);
  void Regularize_L2(Model* model);
  void Regularize_ElasticNet(Model* model);

  real_t learning_rate_;
  real_t regu_lambda_1_;
  real_t regu_lambda_2_;
  std::string regu_type_;  /* l1, l2, elastic_net, or none */

 private:
  DISALLOW_COPY_AND_ASSIGN(Updater);
};

//-----------------;-------------------------------------------------------------
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
