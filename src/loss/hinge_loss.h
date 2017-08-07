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
Author: Chao Ma (mctt90@gmail.com)
This file defines the HingeLoss class.
*/

#ifndef XLEARN_LOSS_HINGE_LOSS_H_
#define XLEARN_LOSS_HINGE_LOSS_H_

#include "src/base/common.h"
#include "src/loss/loss.h"

namespace xLearn {

//------------------------------------------------------------------------------
// HingeLoss is used for classification tasks, which
// has the following form:
// tmp = pred * y
// loss = sum_1_n(tmp < 1 ? 1 - tmp : 0)
//------------------------------------------------------------------------------
class HingeLoss : public Loss {
 public:
  // Constructor and Desstructor
  HingeLoss() { };
  ~HingeLoss() { }

  // Given predictions and labels, return loss value.
  real_t Evalute(const std::vector<real_t>& pred,
                 const std::vector<real_t>& label);

  // Given data sample and current model, calculate gradient
  // and update model.
  void CalcGrad(const DMatrix* data_matrix,
                Model* model,
                Updater* updater);

 private:
  DISALLOW_COPY_AND_ASSIGN(HingeLoss);
};

} // namespace xLearn

#endif // XLEARN_LOSS_HINGE_LOSS_H_
