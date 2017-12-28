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
This file defines the SquaredLoss class.
*/

#ifndef XLEARN_LOSS_SQUARED_LOSS_H_
#define XLEARN_LOSS_SQUARED_LOSS_H_

#include "src/base/common.h"
#include "src/loss/loss.h"

namespace xLearn {

//------------------------------------------------------------------------------
// SquaredLoss is used for regression tasks in machine learning, which
// has the following form:
// loss = sum_all_example( (y - pred) ^ 2 )
//------------------------------------------------------------------------------
class SquaredLoss : public Loss {
 public:
  // Constructor and Desstructor
  SquaredLoss() { };
  ~SquaredLoss() { }

  // Given predictions and labels, accumulate loss value.
  void Evalute(const std::vector<real_t>& pred,
                 const std::vector<real_t>& label);

  // Given data sample and current model, calculate gradient
  // and update current model parameters.
  // This function will also accumulate the loss value.
  void CalcGrad(const DMatrix* data_matrix, Model& model);

  // Given data sample and current model, calculate gradient.
  // Note that this method doesn't update local model, and the
  // gradient will be pushed to the parameter server, which is 
  // used for distributed computation.
  void CalcGradDist(const DMatrix* data_matrix,
                    Model& model,
                    std::vector<real_t>& grad);

  // Return current loss type
  std::string loss_type() { return "mse_loss"; }

 private:
  DISALLOW_COPY_AND_ASSIGN(SquaredLoss);
};

} // namespace xLearn

#endif // XLEARN_LOSS_SQUARED_LOSS_H_
