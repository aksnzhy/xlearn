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

This file tests model_parameters.h
*/

#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "src/data/model_parameters.h"
#include "src/data/hyper_parameters.h"

namespace xLearn {

// Init hyper-parameters
HyperParam Init() {
  HyperParam hyper_param;
  hyper_param.score_func = "ffm";
  hyper_param.loss_func = "squared";
  hyper_param.num_feature = 10;
  hyper_param.num_K = 8;
  hyper_param.num_field = 10;
  hyper_param.model_file = "/tmp/test_model.bin";
  return hyper_param;
}

TEST(MODEL_TEST, Init) {
  // Init model using gaussion.
  HyperParam hyper_param = Init();
  Model model_ffm;
  model_ffm.Initialize(hyper_param.score_func,
                    hyper_param.loss_func,
                    hyper_param.num_feature,
                    hyper_param.num_field,
                    hyper_param.num_K);
  real_t* w = model_ffm.GetParameter_w();
  index_t param_num_w = hyper_param.num_feature +
                      hyper_param.num_feature *
                      hyper_param.num_field *
                      hyper_param.num_K;
  EXPECT_EQ(param_num_w, model_ffm.GetNumParameter_w());
  for (index_t i = 0; i < model_ffm.GetNumFeature(); ++i) {
    EXPECT_FLOAT_EQ(w[i], 0.0);
  }
}

TEST(MODEL_TEST, Save_and_Load) {
  // Init model (set all parameters to zero)
  HyperParam hyper_param = Init();
  Model model_ffm;
  model_ffm.Initialize(hyper_param.score_func,
                    hyper_param.loss_func,
                    hyper_param.num_feature,
                    hyper_param.num_field,
                    hyper_param.num_K);
  real_t* w = model_ffm.GetParameter_w();
  index_t w_len = model_ffm.GetNumParameter_w();
  for (int i = 0; i < w_len; ++i) {
    w[i] = 2.5;
  }
  model_ffm.Serialize(hyper_param.model_file);
  Model new_model(hyper_param.model_file);
  w = new_model.GetParameter_w();
  w_len = new_model.GetNumParameter_w();
  index_t param_num_w = hyper_param.num_feature +
                      hyper_param.num_feature *
                      hyper_param.num_field *
                      hyper_param.num_K;
  EXPECT_EQ(w_len, param_num_w);
  EXPECT_EQ(hyper_param.score_func, new_model.GetScoreFunction());
  EXPECT_EQ(hyper_param.loss_func, new_model.GetLossFunction());
  EXPECT_EQ(hyper_param.num_K, new_model.GetNumK());
  EXPECT_EQ(hyper_param.num_feature, new_model.GetNumFeature());
  EXPECT_EQ(hyper_param.num_field, new_model.GetNumField());

  for (int i = 0; i < w_len; ++i) {
    EXPECT_FLOAT_EQ(w[i], 2.5);
  }
}

}   // namespace xLearn
