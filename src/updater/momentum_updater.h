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

This files defines the Momentum class.
*/

#ifndef XLEARN_UPDATER_MOMENTUM_UPDATER_H_
#define XLEARN_UPDATER_MOMENTUM_UPDATER_H_

#include <vector>

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/updater/updater.h"

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
// [ cache = decay_rate * cache  + gradient ]
// [ w -= learning_rate * cache ]
// The momentum term 'decay_rate' is usually set to 0.9 or a similar value.
//------------------------------------------------------------------------------
class Momentum : public Updater {
 public:
  // Constructor and Destructor
  Momentum() {  }
  ~Momentum() {  }

  // This function need to be invoked before using this class
  void Initialize(real_t learning_rate,
              real_t regu_lambda,
              real_t decay_rate,
              index_t param_num_w);

  // Momentum updater
  void Update(const index_t idx,
              const real_t grad,
              real_t* w);

  // Update a continous space of model using SSE/AVX
  void BatchUpdate(__MX _w,
                   __MX _grad,
                   index_t idx,
                   real_t* w);

 protected:
  real_t decay_rate_;
  __MX _decay_rate;
  real_t* cache_;

 private:
  DISALLOW_COPY_AND_ASSIGN(Momentum);
};

} // namespace xLearn

#endif // XLEARN_UPDATER_MOMENTUM_UPDATER_H_
