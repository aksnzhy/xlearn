//------------------------------------------------------------------------------
// Copyright (c) 2018 by contributors. All Rights Reserved.
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
This file tests the SquaredLoss class.
*/

#include "gtest/gtest.h"

#include <vector>

#include "src/loss/squared_loss.h"
#include "src/score/fm_score.h"

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
  SquaredLoss loss;
  Score* score = new FMScore;
  size_t threadNumber = std::thread::hardware_concurrency();
  ThreadPool* pool = new ThreadPool(threadNumber);
  loss.Initialize(score, pool);
  loss.Evalute(pred, label);
  real_t val = loss.GetLoss();
  EXPECT_FLOAT_EQ(val, 14.25);
}

} // namespace xLearn
