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
index_t K = 10;
index_t Kfeat = 3;
index_t kLength = Kfeat + Kfeat*K;

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
    param.num_field = 3;
    param.num_K = 10;
  }
};

TEST_F(FM_TEST, calc_score) {
  SparseRow row(Kfeat);
  std::vector<real_t> w(kLength, 1.0);
  // Init SparseRow
  for (index_t i = 0; i < Kfeat; ++i) {
    row.idx[i] = i;
    row.X[i] = 2.0;
  }
  HyperParam hyper_param;
  hyper_param.num_feature = Kfeat;
  hyper_param.num_K = K;
  FMScore score;
  score.Initialize(hyper_param.num_feature,
                   hyper_param.num_K);
  real_t val = score.CalcScore(&row, &w);
  // 6 + 10*4*3 = 126
  EXPECT_FLOAT_EQ(val, 126.0);
}

TEST_F(FM_TEST, calc_grad) {
  // Reset hyper parameters
  K = 24;
  Kfeat = 100;
  kLength = Kfeat + Kfeat*K;
  // Create SparseRow
  SparseRow row(Kfeat+1);
  for (index_t i = 0; i < Kfeat; ++i) {
    row.idx[i] = i;
    row.X[i] = 2.0;
  }
  // Create model
  std::vector<real_t> w(kLength, 3.0);
  // Create updater
  Updater* updater = new Updater();
  HyperParam hyper_param;
  hyper_param.learning_rate = 0.1;
  updater->Initialize(hyper_param);
  // Create score function
  FMScore score;
  hyper_param.num_feature = Kfeat;
  hyper_param.num_K = K;
  score.Initialize(hyper_param.num_feature,
                   hyper_param.num_K);
  score.CalcGrad(&row, w, 1.0, updater);
  // Test
  for (index_t i = 0; i < Kfeat; ++i) {
    EXPECT_FLOAT_EQ(w[i], 2.8);
  }
  for (index_t i = Kfeat; i < kLength; ++i) {
    EXPECT_FLOAT_EQ(w[i], -115.8);
  }
}

} // namespace xLearn
