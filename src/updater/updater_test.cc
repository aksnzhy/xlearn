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

This file tests a set of updaters.
*/

#include "gtest/gtest.h"

#include <iostream>

#include "src/base/common.h"

#include "src/updater/updater.h"
#include "src/updater/adadelta_updater.h"
#include "src/updater/adagrad_updater.h"
#include "src/updater/momentum_updater.h"
#include "src/updater/rmsprop_updater.h"

namespace xLearn {

HyperParam param;
const int kLength = _MMX_INCREMENT * 1000000;

class UpdaterTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.num_param = kLength;
    param.loss_func = "sqaured";
    param.score_func = "linear";
    param.num_feature = 100;
    param.num_field = 10;
    param.num_K = 8;
  }
};

TEST_F(UpdaterTest, update_func) {
  Model model;
  model.Initialize(param.num_param,
                param.score_func,
                param.loss_func,
                param.num_feature,
                param.num_field,
                param.num_K,
                false);
  std::vector<real_t> grad_vec(kLength, 2.0);
  std::vector<real_t>* w = model.GetParameter();
  Updater updater;
  updater.Initialize(param.learning_rate,
                  param.regu_lambda,
                  0,
                  0,
                  param.num_param);
  for (int n = 0; n < 3; ++n) {
    for (int i = 0; i < kLength; ++i) {
      updater.Update(i, grad_vec[i], *w);
    }
  }
  for (int i = 0; i < kLength; ++i) {
    EXPECT_FLOAT_EQ((*w)[i], (real_t)(-0.6));
  }
}

TEST_F(UpdaterTest, batch_update_func) {
  std::vector<real_t> K(kLength, 0.0);
  std::vector<real_t> grad_vec(kLength, 2.0);
  Updater updater;
  updater.Initialize(param.learning_rate,
                  param.regu_lambda,
                  0,
                  0,
                  param.num_param);
  for (int n = 0; n < 3; ++n) {
    updater.BatchUpdate(grad_vec, 0, K);
  }
  for (int i = 0; i < kLength; ++i) {
    EXPECT_FLOAT_EQ(K[i], (real_t)(-0.6));
  }
}

Updater* CreateUpdater(const char* format_name) {
  return CREATE_UPDATER(format_name);
}

TEST(UPDATER_TEST, CreateUpdater) {
  EXPECT_TRUE(CreateUpdater("sgd") != NULL);
  EXPECT_TRUE(CreateUpdater("adadelta") != NULL);
  EXPECT_TRUE(CreateUpdater("adagrad") != NULL);
  EXPECT_TRUE(CreateUpdater("momentum") != NULL);
  EXPECT_TRUE(CreateUpdater("rmsprop") != NULL);
  EXPECT_TRUE(CreateUpdater("Unknow_Updater") == NULL);
}

} // namespace xLearn
