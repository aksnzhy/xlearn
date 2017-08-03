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

namespace xLearn {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_loss_registry, Loss);

// Given data sample and current model, return predictions.
void Loss::Predict(const DMatrix* data_matrix,
                   Model* model,
                   std::vector<real>& pred) {
  CHECK_NOTNULL(data_matrix);
  CHECK_NE(pred.empty(), true);
  CHECK_EQ(pred.size(), data_matrix->row_len);
  std::vector<real_t>* w = model->GetParameter();
  for (size_t i = 0; i < data_matrix->row_len; ++i) {
    SparseRow* row = data_matrix->row[i];
    pred[i] = score_func_->CalcScore(row, w);
  }
}

} // xLearn
