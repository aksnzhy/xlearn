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
  std::vector<real_t>* para = model_ffm.GetParameter();
  index_t param_num = hyper_param.num_feature +
                      hyper_param.num_feature *
                      hyper_param.num_field *
                      hyper_param.num_K;
  EXPECT_EQ(para->size(), param_num);
  EXPECT_EQ(model_ffm.GetNumParameter(), param_num);
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
  model_ffm.Serialize(hyper_param.model_file);
  Model new_model(hyper_param.model_file);
  index_t param_num = hyper_param.num_feature +
                      hyper_param.num_feature *
                      hyper_param.num_field *
                      hyper_param.num_K;
  std::vector<real_t>* para = new_model.GetParameter();
  EXPECT_EQ(para->size(), param_num);
  EXPECT_EQ(new_model.GetNumParameter(), param_num);
  EXPECT_EQ(new_model.GetNumField(), hyper_param.num_field);
  EXPECT_EQ(new_model.GetNumK(), hyper_param.num_K);
}

}   // namespace xLearn
