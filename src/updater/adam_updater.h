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

This file defines the AdamUpdater class.
*/

#ifndef XLEARN_UPDATER_ADAM_UPDATER_H_
#define XLEARN_UPDATER_ADAM_UPDATER_H_

#include <vector>

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/updater/updater.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Adaptive Moment Estimation (Adam) is another method that computes adaptive
// learning rates for each parameter. In addition to sotring an exponential
// decaying average of past sqaured gradients Vt like Adadelta and RMSprop,
// Adam also keeps an exponentially decaying of past gradients Mt, similar
// to momentum. as shown in the following form:
// [ m = beta1 * m + (1 - beta1) * dx ]
// [ v = beta2 * v + (1 - beta2) * (dx ^ 2) ]
// [ w -= learning_rate * m / (sqrt(v) + 1e-7) ]
//------------------------------------------------------------------------------
class Adam : public Updater {
 public:
  // Constructor and Desstructor
  Adam() {  }
  ~Adam() {  }

  // This function needs to be invoked before using this class.
  void Initialize(const HyperParam& hyper_param);

  // Adam updater
  void Update(const index_t id,
              const real_t grad,
              std::vector<real_t>& param);

  // Update a continuous space of model parameters using sse/avx.
  void BatchUpdate(const std::vector<real_t>& value,
                   const index_t start_id,
                   std::vector<real_t>& param);

 protected:
  real_t beta1_;
  real_t beta2_;
  int count_num_;
  std::vector<real_t> m_;
  std::vector<real_t> v_;

 private:
  DISALLOW_COPY_AND_ASSIGN(Adam);
};

} // namespace xLearn

#endif // XLEARN_UPDATER_ADAM_UPDATER_H_
