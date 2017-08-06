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

This file tests the AbsLoss class.
*/

#include "gtest/gtest.h"

#include <vector>

#include "src/loss/absolute_loss.h"

namespace xLearn {

index_t kLine = 10;

TEST(SQUARED_LOSS, Evalute) {
  // Create pred vector
  std::vector<real_t> pred(kLine);
  for (int i = 0; i < pred.size(); ++i) {
    pred[i] = i;
  }
  // Create label vector
  std::vector<real_t> label(kLine);
  for (int i = 0; i < label.size(); ++i) {
    label[i] = 2*i;
  }
  // Create loss
  AbsLoss loss;
  real_t val = loss.Evalute(pred, label);
  EXPECT_EQ(val, 45.0);
}

} // namespace xLearn
