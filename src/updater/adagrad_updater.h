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

This file defines the AdaGradUpdater class.
*/

#ifndef XLEAR_UPDATER_ADAGRAD_UPDATER_H_
#define XLEAR_UPDATER_ADAGRAD_UPDATER_H_

#include <vector>

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/updater/updater.h"

namespace xLearn {

//------------------------------------------------------------------------------
// AdaGrad is an algorithm for gradient-based optimization that does just
// this: It adapts the learning rate to the parameters, performing larger
// updates for infrequent and smaller updates for frequent parameters. For
// this reason, it is well-suited for dealing with sparse data. AdaGrad uses
// the following update:
// [ cache += gradient ** 2 ]
// [ w += -learning_rate * gradient/ (sqrt(cache) + 1e-7) ]
//------------------------------------------------------------------------------
class AdaGrad : public Updater {
 public:
  // Constructor and Desstructor
  AdaGrad() {  }
  ~AdaGrad() {  }

  // This function neede to be invoked before using this class
  void Initialize(real_t learning_rate,
               real_t regu_lambda,
               real_t decay_rate = 0,
               index_t num_param_w = 0);

  // AdaGrad updater
  void Update(const index_t idx,
              const real_t grad,
              real_t* w);

  // Update a continous space of model using SSE/AVX
  void BatchUpdate(__MX _w,
                   __MX _grad,
                   index_t idx,
                   real_t* w);

 protected:
  real_t* cache_;

 private:
  DISALLOW_COPY_AND_ASSIGN(AdaGrad);
};

} // namespace xLearn

#endif // XLEAR_UPDATER_ADAGRAD_UPDATER_H_
