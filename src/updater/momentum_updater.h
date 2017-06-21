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

This files defines the MomentumUpdater class.
*/

#ifndef XLEARN_UPDATE_MOMENTUM_UPDATER_H_
#define XLEARN_UPDATE_MOMENTUM_UPDATER_H_

#include <vector>

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/update/updater.h"

namespace xLearn {

//------------------------------------------------------------------------------
// SGD has trouble navigating ravines, i.e. areas where the surface curves
// much more steeply in one dimension than in another, which are common
// around local optimal. In these scenarios, SGD oscillates across the slopes
// of the ravine while only making hesitant progress along the bottom towards
// the local optimum.
// Momentum is a method that helps accelerate SGD in the relevant direction
// and dampens oscillations. It does this by a fraction 'velocity' (v) of the
// update vector of the past time step to the current update vector:
// [ v = rho * v  + gradient ]
// [ w -= learning_rate * v ]
// The momentum term 'rho' is usually set to 0.9 or a similar value.
//------------------------------------------------------------------------------
class Momentum : public Updater {
 public:
  // Constructor and Destructor
  Momentum() {  }
  ~Momentum() {  }

  // This function need to be invoked before using this class.
  void Initialize(const HyperParam& hyper_param);

  // Momentum updater
  void Update(const index_t id,
              const real_t grad,
              std::vector<real_t>& param);

  // Update a continuous space of model parameters using SSE/AVX.
  void BatchUpdate(const std::vector<real_t>& value,
                   const index_t start_id,
                   std::vector<real_t>& param);

 protected:
  real_t rho_;
  std::vector<real_t> v_;

 private:
  DISALLOW_COPY_AND_ASSIGN(Momentum);
};

} // namespace xLearn

#endif // XLEARN_UPDATE_MOMENTUM_UPDATER_H_
