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

This file tests the LinearScore class.
*/

#include "gtest/gtest.h"

#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"

#include "src/score/score_function.h"
#include "src/score/linear_score.h"

namespace xLearn {

index_t kLength = 100;

TEST(LINEAR_TEST, calc_score) {
  SparseRow row(kLength);
  std::vector<real_t> w(kLength, 3.0);
  // Init SparseRow
  for (index_t i = 0; i < kLength; ++i) {
    row.idx[i] = i;
    row.X[i] = 2.0;
  }
  LinearScore score;
  real_t val = score.CalcScore(&row, &w);
  EXPECT_FLOAT_EQ(val, 600.0);
}

TEST(LINEAR_TEST, calc_grad) {

}

} // namespace xLearn
