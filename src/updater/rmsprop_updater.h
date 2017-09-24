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

This file defines the RMSProp class.
*/

#ifndef XLEAR_UPDATER_RMSPROP_UPDATER_H_
#define XLEAR_UPDATER_RMSPROP_UPDATER_H_

#include <vector>

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/updater/updater.h"

namespace xLearn {

//------------------------------------------------------------------------------
// RMSProp is an unpublished, adaptive learning rate method proposed by
// Geoff Hinton in Lecture 6e of his Coursera Class. RMSProp and AdaDelta
// have both been developed independently around the same time stemming from
// the need to resolve Adagrad's radically diminishing learning rate. RMSProp
// in fact is identical to the first update vector of AdaDelta, as shown below:
// [ cache = decay_rate * cache + (1 - decay_rate) * dx ^ 2 ]
// [ w -= learning_rate * dx / (sqrt(cache) + 1e-7) ]
//------------------------------------------------------------------------------
class RMSProp : public Updater {
 public:
  // Constructor and Destructor
  RMSProp() {  }
  ~RMSProp() {  }

  // This function needs to be invoked before using this class
  void Initialize(real_t learning_rate,
              real_t regu_lambda,
              real_t decay_rate,
              index_t param_num_w);

  // RMSProp update
  void Update(const index_t idx,
              const real_t grad,
              real_t* w);

  // Update a continous space of model using sse/avx
  void BatchUpdate(__MX _w,
                   __MX _grad,
                   index_t idx,
                   real_t* w);

 protected:
  real_t decay_rate_;
  __MX _decay_rate;
  real_t* cache_;

 private:
  DISALLOW_COPY_AND_ASSIGN(RMSProp);
};

}  // namespace xLearn

#endif  // XLEAR_UPDATER_RMSPROP_UPDATER_H_
