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

This file tests the FFMScore class.
*/

#include "gtest/gtest.h"

#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"

#include "src/score/score_function.h"
#include "src/score/ffm_score.h"

namespace xLearn {

index_t K = 24;
index_t Kfeat = 3;
index_t kfield = 3;
index_t kLength = Kfeat + Kfeat*kfield*K;

TEST(FFM_TEST, calc_score) {
  SparseRow row(Kfeat, true);
  std::vector<real_t> w(kLength, 1.0);
  // Init SparseRow
  for (index_t i = 0; i < Kfeat; ++i) {
    row.idx[i] = i;
    row.field[i] = i;
    row.X[i] = 2.0;
  }
  HyperParam hyper_param;
  hyper_param.num_feature = Kfeat;
  hyper_param.num_K = K;
  hyper_param.num_field = kfield;
  FFMScore score;
  score.Initialize(hyper_param);
  real_t val = score.CalcScore(&row, &w);
  // 6 + 24*4*3 = 294.0
  EXPECT_FLOAT_EQ(val, 294.0);
}

TEST(FFM_TEST, calc_grad) {
  // Reset hyper parameters
  K = 24;
  Kfeat = 100;
  kfield = 100;
  kLength = Kfeat + Kfeat*kfield*K;
  // Create SparseRow
  SparseRow row(Kfeat, true);
  for (index_t i = 0; i < Kfeat; ++i) {
    row.idx[i] = i;
    row.X[i] = 2.0;
    row.field[i] = i;
  }
  // Create model
  std::vector<real_t> w(kLength, 3.0);
  // Create updater
  Updater* updater = new Updater();
  HyperParam hyper_param;
  hyper_param.learning_rate = 0.1;
  updater->Initialize(hyper_param);
  // Create score function
  FFMScore score;
  hyper_param.num_feature = Kfeat;
  hyper_param.num_K = K;
  hyper_param.num_field = kfield;
  score.Initialize(hyper_param);
  score.CalcGrad(&row, w, 1.0, updater);
  // Test
  for (index_t i = 0; i < Kfeat; ++i) {
    EXPECT_FLOAT_EQ(w[i], 2.8);
  }
  for (index_t i = Kfeat; i < kLength; ++i) {
    if (w[i] != 3.0) {
      EXPECT_FLOAT_EQ(w[i], 1.8);
    }
  }
}

} // namespace xLearn
