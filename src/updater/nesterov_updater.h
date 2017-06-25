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

This files defines the Nesterov class.
*/

#ifndef XLEARN_UPDATE_NESTORV_UPDATER_H_
#define XLEARN_UPDATE_NESTORV_UPDATER_H_

#include <vector>

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/updater/updater.h"

namespace xLearn {

//------------------------------------------------------------------------------
// A ball that rolls down a hill, blindly following the slope, is highly
// unsatisfactory. We'd like to have a smarter ball, a ball that has a notion
// of where it is going so that it knows to slow down before the hill slopes
// up again. Nesterov accelerated gradient (NAG) is a way to give our momentum
// term this kind of prescience. We know that we will use our momentum term
// γ*v_t-1 to move the parameters θ. Computing θ − γ*v_t-1 thus gives us an
// approximation of the next position of the parameters (the gradient is
// missing for the full update), a rough idea where our parameters are going
// to be. We can now effectively look ahead by calculating the gradient not
// w.r.t. to our current parameters θ but w.r.t. the approximate future
// position of our parameters:
// [ old_v = v ]
// [ v = rho * v - learning_rate* gradient ]
// [ x += -rho * old_v + (1+rho) * v]
//------------------------------------------------------------------------------
class Nesterov : public Updater {
 public:
  // Constructor and Destructor
  Nesterov() { }
  ~Nesterov() { }

  // This function need to be invoked before using this class.
  void Initialize(const HyperParam& hyper_param);

  // Nesterov updater
  void Update(const index_t id,
              const real_t grad,
              std::vector<real_t>& param);

  // Update a continuous space of model parameters using sse/avx.
  void BatchUpdate(const std::vector<real_t>& value,
                   const index_t start_id,
                   std::vector<real_t>& param);

 protected:
  real_t rho_;
  std::vector<real_t> v_;

 private:
  DISALLOW_COPY_AND_ASSIGN(Nesterov);
};

} // namespace xLearn

#endif // XLEARN_UPDATE_NESTORV_UPDATER_H_
