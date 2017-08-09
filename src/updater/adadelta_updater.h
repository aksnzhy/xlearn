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

This files defines the AdaDeltaUpdater class.
*/

#ifndef XLEARN_UPDATER_ADADELTA_UPDATER_H_
#define XLEARN_UPDATER_ADADELTA_UPDATER_H_

#include <vector>

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/updater/updater.h"

namespace xLearn {

//------------------------------------------------------------------------------
// AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive,
// monotonically decreasing learning rate. Instead of accumulating all past
// squared gradients, AdaDelta restricts the window of accumulated past
// gradients to some fixed size w.
// [ cache = (1-decay_rate) * (grad^2) + decay_rate * cache ]
// [ w -= learning_rate * grad  / sqrt(cache) ]
//------------------------------------------------------------------------------
class AdaDelta : public Updater {
 public:
  // Constructor and Desstructor
  AdaDelta() {  }
  ~AdaDelta() {  }

  // This function needs to be invoked before update.
  void Initialize(const HyperParam& hyper_param);

  // AdaDelta update
  void Update(const index_t id,
              const real_t grad,
              std::vector<real_t>& param);

  // Update a continous space of model parameters using sse/avx.
  void BatchUpdate(const std::vector<real_t>& value,
                   const index_t start_id,
                   std::vector<real_t>& param);
                   
 protected:
  std::vector<real_t> cache_;
  real_t decay_rate_;

 private:
  DISALLOW_COPY_AND_ASSIGN(AdaDelta);
};

} // namespace xLearn

#endif // XLEARN_UPDATER_ADADELTA_UPDATER_H_
