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
This file is the implementation of CrossEntropyLoss class.
*/

#include "src/loss/cross_entropy_loss.h"

namespace xLearn {

// Given predictions and labels, return cross-entropy loss value.
real_t CrossEntropyLoss::Evalute(const std::vector<real_t>& pred,
                                 const std::vector<real_t>& label) {
  CHECK_EQ(pred.empty(), false);
  CHECK_EQ(pred.size(), label.size());
  real_t val = 0.0;
  for (size_t i = 0; i < pred.size(); ++i) {
    real_t y = label[i] > 0 ? 1.0 : -1.0;
    val += log(1.0 + exp(-y*pred[i]));
  }
  return val;
}

// Given data sample and current model, return gradients.
void CrossEntropyLoss::CalcGrad(const DMatrix* data_matrix,
                                Model* model,
                                Updater* updater) {

}

} // namespace xLearn
