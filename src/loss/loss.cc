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
This file is the implementation of the base Loss class.
*/

#include "src/loss/loss.h"
#include "src/loss/squared_loss.h"
#include "src/loss/hinge_loss.h"
#include "src/loss/cross_entropy_loss.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_loss_registry, Loss);
REGISTER_LOSS("squared", SquaredLoss);
REGISTER_LOSS("hinge", HingeLoss);
REGISTER_LOSS("cross-entropy", CrossEntropyLoss);

// Given data sample and current model, return predictions.
void Loss::Predict(const DMatrix* matrix,
                   Model& model,
                   std::vector<real_t>& pred) {
  CHECK_NOTNULL(matrix);
  CHECK_NE(pred.empty(), true);
  CHECK_EQ(pred.size(), matrix->row_length);
  for (size_t i = 0; i < matrix->row_length; ++i) {
    SparseRow* row = matrix->row[i];
    pred[i] = score_func_->CalcScore(row, model, matrix->norm[i]);
  }
}

} // xLearn
