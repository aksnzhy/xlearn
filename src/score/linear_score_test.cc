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
This file tests the LinearScore class.
*/

#include "gtest/gtest.h"

#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"

#include "src/score/score_function.h"
#include "src/score/linear_score.h"

namespace xLearn {

HyperParam param;
const int kLength = 100;

class LinearScoreTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.num_param = kLength;
    param.loss_func = "squared";
    param.score_func = "linear";
    param.num_feature = kLength;
  }
};

TEST_F(LinearScoreTest, calc_score) {
  SparseRow row(kLength);
  Model model;
  model.Initialize(param.score_func,
                param.loss_func,
                param.num_feature,
                0, 0, 2);
  real_t* w = model.GetParameter_w();
  index_t num_w = model.GetNumParameter_w();
  for (index_t i = 0; i < num_w; ++i) {
    w[i] = 3.0;
  }
  model.GetParameter_b()[0] = 0.0;
  // Init SparseRow
  for (index_t i = 0; i < kLength; ++i) {
    row[i].feat_id = i;
    row[i].feat_val = 2.0;
  }
  LinearScore score;
  real_t val = score.CalcScore(&row, model);
  EXPECT_FLOAT_EQ(val, 600.0);
}

TEST_F(LinearScoreTest, calc_score_overflow) {
  SparseRow row(2*kLength);
  Model model;
  model.Initialize(param.score_func,
                param.loss_func,
                param.num_feature,
                0, 0, 2);
  real_t* w = model.GetParameter_w();
  index_t num_w = model.GetNumParameter_w();
  for (index_t i = 0; i < num_w; ++i) {
    w[i] = 3.0;
  }
  model.GetParameter_b()[0] = 0.0;
  // Init SparseRow
  for (index_t i = 0; i < 2*kLength; ++i) {
    row[i].feat_id = i;
    row[i].feat_val = 2.0;
  }
  LinearScore score;
  real_t val = score.CalcScore(&row, model);
  EXPECT_FLOAT_EQ(val, 600.0);
}

} // namespace xLearn
