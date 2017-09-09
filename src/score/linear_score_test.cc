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
#include "src/updater/updater.h"

#include "src/score/score_function.h"
#include "src/score/linear_score.h"

namespace xLearn {

HyperParam param;
index_t kLength = 100;

class LinearScoreTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.decay_rate_1 = 0.91;
    param.num_param = kLength;
    param.loss_func = "sqaured";
    param.score_func = "linear";
    param.num_feature = 100;
    param.num_field = 10;
    param.num_K = 8;
  }
};

TEST_F(LINEAR_TEST, calc_score) {
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

TEST_F(LINEAR_TEST, calc_grad) {
  // Create SparseRow
  SparseRow row(kLength);
  for (index_t i = 0; i < kLength; ++i) {
    row.idx[i] = i;
    row.X[i] = 2.0;
  }
  // Create model
  std::vector<real_t> w(kLength, 3.0);
  // Create updater
  Updater* updater = new Updater();
  updater->Initialize(param.learning_rate,
                  param.regu_lambda,
                  0,
                  0,
                  param.num_param);
  // Create score function
  LinearScore score;
  score.CalcGrad(&row, w, 1.0, updater);
  // Test
  for (index_t i = 0; i < kLength; ++i) {
    EXPECT_FLOAT_EQ(w[i], 2.8);
  }
}

} // namespace xLearn
