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

This file tests the FMScore class.
*/

#include "gtest/gtest.h"

#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"

#include "src/score/score_function.h"
#include "src/score/fm_score.h"

namespace xLearn {

HyperParam param;

class FMScoreTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.loss_func = "sqaured";
    param.score_func = "fm";
    param.num_feature = 3;
    param.num_K = 20;
  }
};

TEST_F(FMScoreTest, calc_score) {
  // Init SparseRow
  SparseRow row(param.num_feature);
  for (index_t i = 0; i < param.num_feature; ++i) {
    row[i].feat_id = i;
    row[i].feat_val = 2.0;
  }
  // Init model
  Model model;
  model.Initialize(param.score_func,
                param.loss_func,
                param.num_feature,
                param.num_field,
                param.num_K);
  real_t* w = model.GetParameter_w();
  index_t num_w = model.GetNumParameter_w();
  for (index_t i = 0; i < num_w; ++i) {
    w[i] = 1.0;
  }
  real_t* v = model.GetParameter_v();
  index_t num_v = model.GetNumParameter_v();
  for (index_t i = 0; i < num_v; ++i) {
    v[i] = 1.0;
  }
  model.GetParameter_b()[0] = 0.0;
  FMScore score;
  real_t val = score.CalcScore(&row, model);
  // 6 + 20*4*3 = 246
  EXPECT_FLOAT_EQ(val, 246);
}

} // namespace xLearn
