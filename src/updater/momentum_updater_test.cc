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

This file tests the Momentum class.
*/

#include "gtest/gtest.h"

#include "src/base/common.h"

#include "src/updater/updater.h"
#include "src/updater/momentum_updater.h"

namespace xLearn {

HyperParam param;

class MomentumTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.decay_rate = 1.0;
    param.loss_func = "sqaured";
    param.score_func = "linear";
    param.num_feature = 100;
    param.num_field = 10;
    param.num_K = 8;
  }
};

TEST_F(MomentumTest, update_func) {
  Model model;
  model.Initialize(param.score_func,
                param.loss_func,
                param.num_feature,
                param.num_field,
                param.num_K);
  index_t length = model.GetNumParameter_w();
  real_t* w = model.GetParameter_w();
  Momentum updater;
  updater.Initialize(param.learning_rate,
                  param.regu_lambda,
                  param.decay_rate,
                  length);
  for (int n = 0; n < 3; ++n) {
    for (int i = 0; i < length; ++i) {
      updater.Update(i, 1.0, w);
    }
  }
  for (int i = 0; i < length; ++i) {
    EXPECT_FLOAT_EQ(w[i], -0.6);
  }
}

TEST_F(MomentumTest, batch_update_func) {
  __MX _grad = _MMX_SET1_PS(1.0);
  real_t *w = new real_t[_MMX_INCREMENT];
  for (int i = 0; i < _MMX_INCREMENT; ++i) {
    w[i] = 0.0;
  }
  Momentum updater;
  updater.Initialize(param.learning_rate,
                  param.regu_lambda,
                  param.decay_rate,
                  _MMX_INCREMENT);
  for (int n = 0; n < 3; ++n) {
    __MX _w = _MMX_LOAD_PS(w);
    updater.BatchUpdate(_w, _grad, 0, w);
  }
  for (int i = 0; i < _MMX_INCREMENT; ++i) {
    EXPECT_FLOAT_EQ(w[i], -0.6);
  }
}

} // namespace xLearn
