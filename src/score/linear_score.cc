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
This file is the implementation of LinearScore class.
*/

#include "src/score/linear_score.h"
#include "src/score/fm_score.h"

namespace xLearn {

// y = wTx (bias is added in w and x automitically)
real_t LinearScore::CalcScore(const SparseRow* row,
                              const std::vector<real_t>* w) {
  real_t score = 0.0;
  for (index_t j = 0; j < row->column_len; ++j) {
    index_t idx = row->idx[j];
    score += (*w)[idx] * row->X[j];
  }
  return score;
}

} // namespace xLearn
