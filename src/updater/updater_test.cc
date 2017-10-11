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

class UpdaterTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.loss_func = "sqaured";
    param.score_func = "linear";
    param.num_feature = 100;
    param.num_field = 10;
    param.num_K = 8;
  }
};

TEST_F(UpdaterTest, update_func) {
  Model model;
  model.Initialize(param.score_func,
                param.loss_func,
                param.num_feature,
                param.num_field,
                param.num_K);
  Updater updater;
  updater.Initialize(param.learning_rate,
                     param.regu_lambda);
  index_t length = model.GetNumParameter_w();
  real_t* w = model.GetParameter_w();
  for (int n = 0; n < 3; ++n) {
    for (int i = 0; i < length; ++i) {
      updater.Update(i, 2.0, w);
    }
  }
  for (int i = 0; i < length; ++i) {
    EXPECT_FLOAT_EQ(w[i], -0.6);
  }
}

TEST_F(UpdaterTest, batch_update_func) {
  Updater updater;
  updater.Initialize(param.learning_rate,
                  param.regu_lambda);
  __MX _grad = _MMX_SET1_PS(2.0);
  real_t *w = new real_t[_MMX_INCREMENT];
  for (int i = 0; i < _MMX_INCREMENT; ++i) {
    w[i] = 0.0;
  }
  for (int n = 0; n < 3; ++n) {
    __MX _w = _MMX_LOAD_PS(w);
    updater.BatchUpdate(_w, _grad, 0, w);
  }
  for (int i = 0; i < _MMX_INCREMENT; ++i) {
    EXPECT_FLOAT_EQ(w[i], -0.6);
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
