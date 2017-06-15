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

#ifndef F2M_UPDATE_ADADELTA_UPDATER_H_
#define F2M_UPDATE_ADADELTA_UPDATER_H_

#include <vector>

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/update/updater.h"

namespace f2m {

//------------------------------------------------------------------------------
// AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive,
// monotonically decreasing learning rate. Instead of accumulating all past
// squared gradients, AdaDelta restricts the window of accumulated past
// gradients to some fixed size w.
//------------------------------------------------------------------------------
class AdaDeltaUpdater : public Updater {
 public:
  // Constructor and Desstructor
  AdaDeltaUpdater() {  }
  ~AdaDeltaUpdater() {  }

  // This function needs to be invoked before update.
  void Initialize(const HyperParam& hyper_param);

  // AdaDelta update
  void Update(index_t key, real_t grad, Model* model);

  // Update model parameter in a mini-batch GD
  void BatchUpdate(Gradient* grad, Model* model);

  // Update a continuous model parameter
  void SeqUpdate(std::vector<real_t>& value,
                 index_t start_key,
                 Model* model);

 private:
  real_t decay_rate_;

  DISALLOW_COPY_AND_ASSIGN(AdaDeltaUpdater);
};

} // namespace f2m

#endif // F2M_UPDATE_ADADELTA_UPDATER_H_
