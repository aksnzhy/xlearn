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
#include "src/base/file_util.h"

namespace xLearn {

const uint32 kParameter_num = 2500 * 4; // Assume kParameter_num % 4 == 0
const std::string kFilename = "/tmp/test_model.binary";

//------------------------------------------------------------------------------
// Model test
//------------------------------------------------------------------------------

// Init hyper-parameters
HyperParam Init() {
  HyperParam hyper_param;
  hyper_param.score_func = "ffm";
  hyper_param.loss_func = "squared";
  hyper_param.num_feature = 10;
  hyper_param.num_K = 8;
  hyper_param.num_field = 10;
  hyper_param.num_param = kParameter_num;
  return hyper_param;
}

TEST(MODEL_TEST, Init) {
  // Init model using gaussion.
  HyperParam hyper_param = Init();
  Model model_ffm;
  model_ffm.Initialize(hyper_param.num_param,
                       hyper_param.score_func,
                       hyper_param.loss_func,
                       hyper_param.num_feature,
                       hyper_param.num_field,
                       hyper_param.num_K);
  std::vector<real_t>* para = model_ffm.GetParameter();
  EXPECT_EQ(para->size(), kParameter_num);
}

TEST(MODEL_TEST, SaveModel) {
  // Init model (set all parameters to zero)
  HyperParam hyper_param = Init();
  Model model_ffm;
  model_ffm.Initialize(hyper_param.num_param,
                       hyper_param.score_func,
                       hyper_param.loss_func,
                       hyper_param.num_feature,
                       hyper_param.num_field,
                       hyper_param.num_K,
                       false);
  model_ffm.SaveModel(kFilename);
}

TEST(MODEL_TEST, LoadModel) {
  // Init model with gaussion distribution.
  HyperParam hyper_param = Init();
  Model model_ffm;
  model_ffm.Initialize(hyper_param.num_param,
                       hyper_param.score_func,
                       hyper_param.loss_func,
                       hyper_param.num_feature,
                       hyper_param.num_field,
                       hyper_param.num_K);
  // parameters become 0
  model_ffm.LoadModel(kFilename);
  EXPECT_EQ(model_ffm.GetScoreFunction(), "ffm");
  EXPECT_EQ(model_ffm.GetLossFunction(), "squared");
  EXPECT_EQ(model_ffm.GetNumFeature(), 10);
  EXPECT_EQ(model_ffm.GetNumK(), 8);
  EXPECT_EQ(model_ffm.GetNumField(), 10);
  std::vector<real_t>* para = model_ffm.GetParameter();
  for (index_t i = 0; i < para->size(); ++i) {
    EXPECT_EQ((*para)[i], (real_t)0.0);
  }
}

TEST(MODEL_TEST, InitModelFromDiskfile) {
  Model model_ffm(kFilename);
  EXPECT_EQ(model_ffm.GetScoreFunction(), "ffm");
  EXPECT_EQ(model_ffm.GetLossFunction(), "squared");
  EXPECT_EQ(model_ffm.GetNumFeature(), 10);
  EXPECT_EQ(model_ffm.GetNumK(), 8);
  EXPECT_EQ(model_ffm.GetNumField(), 10);
  std::vector<real_t>* para = model_ffm.GetParameter();
  for (index_t i = 0; i < para->size(); ++i) {
    EXPECT_EQ((*para)[i], (real_t)0.0);
  }
}

TEST(MODEL_TEST, RemoveFile) {
  Model model;
  model.RemoveModelFile(kFilename.c_str());
}

TEST(MODEL_TEST, SaveweightAndLoadweight) {
  HyperParam hyper_param = Init();
  Model model_ffm;
  model_ffm.Initialize(hyper_param.num_param,
                       hyper_param.score_func,
                       hyper_param.loss_func,
                       hyper_param.num_feature,
                       hyper_param.num_field,
                       hyper_param.num_K,
                       false);
  std::vector<real_t> vec(kParameter_num, 1.0);
  model_ffm.Saveweight(vec);
  for (index_t i = 0; i < vec.size(); ++i) {
    EXPECT_EQ(vec[i], 0);
    vec[i] = 2.0;
  }
  model_ffm.Loadweight(vec);
  std::vector<real_t>* para = model_ffm.GetParameter();
  for (index_t i = 0; i < para->size(); ++i) {
    EXPECT_EQ((*para)[i], 2.0);
  }
}

} // namespace xLearn
