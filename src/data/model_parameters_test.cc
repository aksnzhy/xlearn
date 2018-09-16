//------------------------------------------------------------------------------
// Copyright (c) 2018 by contributors. All Rights Reserved.
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
This file tests model_parameters.h file.
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
  hyper_param.num_feature = 4;
  hyper_param.num_K = 4;
  hyper_param.auxiliary_size = 2;
  hyper_param.num_field = 4;
  hyper_param.model_file = "./test_model.bin";
  return hyper_param;
}

TEST(MODEL_TEST, Init_ffm) {
  HyperParam hyper_param = Init();
  Model model_ffm;
  model_ffm.Initialize(hyper_param.score_func,
                    hyper_param.loss_func,
                    hyper_param.num_feature,
                    hyper_param.num_field,
                    hyper_param.num_K,
                    hyper_param.auxiliary_size);
  real_t* b = model_ffm.GetParameter_b();
  real_t* w = model_ffm.GetParameter_w();
  index_t param_num_w = hyper_param.num_feature * 2;
  real_t* v = model_ffm.GetParameter_v();
  index_t param_num_v = hyper_param.num_feature *
                      hyper_param.num_field *
                      hyper_param.num_K * 2;
  EXPECT_EQ(param_num_w, model_ffm.GetNumParameter_w());
  EXPECT_EQ(param_num_v, model_ffm.GetNumParameter_v());
  EXPECT_EQ(param_num_v+param_num_w+2, model_ffm.GetNumParameter());
  EXPECT_FLOAT_EQ(b[0], 0);
  EXPECT_FLOAT_EQ(b[1], 1.0);
  index_t len = model_ffm.GetNumParameter_w();
  index_t aux_size = hyper_param.auxiliary_size;
  for (index_t i = 0; i < len; i += aux_size) {
    EXPECT_FLOAT_EQ(w[i], 0.0);
    for (index_t j = 1; j < aux_size; ++j) {
      EXPECT_FLOAT_EQ(w[i+j], 1.0);
    }

  }
  len = model_ffm.GetNumParameter_v();
  for (index_t i = 0; i < len; i+=(kAlign*aux_size)) {
    for (index_t j = 1; j < aux_size; ++j) {
      EXPECT_FLOAT_EQ(v[i+kAlign*j], 1.0);
    }
  }
}

TEST(MODEL_TEST, Init_fm) {
  HyperParam hyper_param = Init();
  hyper_param.score_func = "fm";
  Model model_fm;
  model_fm.Initialize(hyper_param.score_func,
                    hyper_param.loss_func,
                    hyper_param.num_feature,
                    hyper_param.num_field,
                    hyper_param.num_K, 
                    hyper_param.auxiliary_size, 
                    0.5);
  real_t* b = model_fm.GetParameter_b();
  real_t* w = model_fm.GetParameter_w();
  index_t param_num_w = hyper_param.num_feature * 2;
  real_t* v = model_fm.GetParameter_v();
  index_t param_num_v = hyper_param.num_feature *
                        hyper_param.num_K * 2;
  EXPECT_EQ(param_num_w, model_fm.GetNumParameter_w());
  EXPECT_EQ(param_num_v, model_fm.GetNumParameter_v());
  EXPECT_EQ(param_num_v+param_num_w+2, model_fm.GetNumParameter());
  EXPECT_FLOAT_EQ(b[0], 0);
  EXPECT_FLOAT_EQ(b[1], 1.0);
  index_t len = model_fm.GetNumParameter_w();
  index_t aux_size = hyper_param.auxiliary_size;
  for (index_t i = 0; i < len; i+=aux_size) {
    EXPECT_FLOAT_EQ(w[i], 0.0);
    for (index_t j = 1; j < aux_size; ++j) {
      EXPECT_FLOAT_EQ(w[i+j], 1.0);
    }
  }
  index_t k_aligned = model_fm.get_aligned_k();
  for (index_t i = k_aligned; i < param_num_v; i+=(aux_size*k_aligned)) {
    EXPECT_FLOAT_EQ(v[i], 1.0);
  }
}

TEST(MODEL_TEST, Init_lr) {
  HyperParam hyper_param = Init();
  hyper_param.score_func = "linear";
  Model model_lr;
  model_lr.Initialize(hyper_param.score_func,
                    hyper_param.loss_func,
                    hyper_param.num_feature,
                    hyper_param.num_field,
                    hyper_param.num_K,
                    hyper_param.auxiliary_size, 
                    0.5);
  real_t* b = model_lr.GetParameter_b();
  real_t* w = model_lr.GetParameter_w();
  index_t param_num_w = hyper_param.num_feature * 2;
  real_t* v = model_lr.GetParameter_v();
  index_t param_num_v = 0;
  EXPECT_EQ(param_num_w, model_lr.GetNumParameter_w());
  EXPECT_EQ(param_num_v, model_lr.GetNumParameter_v());
  EXPECT_EQ(param_num_v+param_num_w+2, model_lr.GetNumParameter());
  EXPECT_FLOAT_EQ(b[0], 0);
  EXPECT_FLOAT_EQ(b[1], 1.0);
  index_t aux_size = hyper_param.auxiliary_size;
  for (index_t i = 0; i < model_lr.GetNumParameter_w(); i+=aux_size) {
    EXPECT_FLOAT_EQ(w[i], 0.0);
    for (index_t j = 1; j < aux_size; ++j) {
      EXPECT_FLOAT_EQ(w[i+j], 1.0);
    }
  }
  EXPECT_EQ(v, nullptr);
}

TEST(MODEL_TEST, Save_and_Load) {
  // Init model
  HyperParam hyper_param = Init();
  Model model_ffm;
  model_ffm.Initialize(hyper_param.score_func,
                    hyper_param.loss_func,
                    hyper_param.num_feature,
                    hyper_param.num_field,
                    hyper_param.num_K, 
                    hyper_param.auxiliary_size);
  real_t* w = model_ffm.GetParameter_w();
  index_t w_len = model_ffm.GetNumParameter_w();
  for (int i = 0; i < w_len; ++i) {
    w[i] = 2.5;
  }
  real_t* v = model_ffm.GetParameter_v();
  index_t v_len = model_ffm.GetNumParameter_v();
  for (int i = 0; i < v_len; ++i) {
    v[i] = 3.5;
  }
  model_ffm.Serialize(hyper_param.model_file);
  Model new_model(hyper_param.model_file);
  real_t* b = new_model.GetParameter_b();
  w = new_model.GetParameter_w();
  w_len = new_model.GetNumParameter_w();
  v = new_model.GetParameter_v();
  v_len = new_model.GetNumParameter_v();
  index_t param_num_w = hyper_param.num_feature * 2;
  index_t param_num_v = hyper_param.num_feature *
                      hyper_param.num_field *
                      hyper_param.num_K * 2;
  EXPECT_EQ(w_len, param_num_w);
  EXPECT_EQ(v_len, param_num_v);
  EXPECT_EQ(param_num_v+param_num_w+2, new_model.GetNumParameter());
  EXPECT_EQ(hyper_param.score_func, new_model.GetScoreFunction());
  EXPECT_EQ(hyper_param.loss_func, new_model.GetLossFunction());
  EXPECT_EQ(hyper_param.num_K, new_model.GetNumK());
  EXPECT_EQ(hyper_param.num_feature, new_model.GetNumFeature());
  EXPECT_EQ(hyper_param.num_field, new_model.GetNumField());
  EXPECT_EQ(hyper_param.auxiliary_size, new_model.GetAuxiliarySize());
  EXPECT_FLOAT_EQ(b[0], 0);
  EXPECT_FLOAT_EQ(b[1], 1.0);
  for (int i = 0; i < w_len; ++i) {
    EXPECT_FLOAT_EQ(w[i], 2.5);
  }
  for (int i = 0; i < v_len; ++i) {
    EXPECT_FLOAT_EQ(v[i], 3.5);
  }
  RemoveFile(hyper_param.model_file.c_str());
}

TEST(MODEL_TEST, SerializeToTXT) {
  HyperParam hyper_param = Init();
  // linear
  hyper_param.score_func = "linear";
  Model model_lr;
  model_lr.Initialize(hyper_param.score_func,
                    hyper_param.loss_func,
                    hyper_param.num_feature,
                    hyper_param.num_field,
                    hyper_param.num_K,
                    hyper_param.auxiliary_size, 
                    0.5);
  model_lr.SerializeToTXT("test_txt.linear");
  // fm
  hyper_param.score_func = "fm";
  Model model_fm;
  model_fm.Initialize(hyper_param.score_func,
                    hyper_param.loss_func,
                    hyper_param.num_feature,
                    hyper_param.num_field,
                    hyper_param.num_K,
                    hyper_param.auxiliary_size, 
                    0.5);
  model_fm.SerializeToTXT("test_txt.fm");
  // ffm
  hyper_param.score_func = "ffm";
  Model model_ffm;
  model_ffm.Initialize(hyper_param.score_func,
                    hyper_param.loss_func,
                    hyper_param.num_feature,
                    hyper_param.num_field,
                    hyper_param.num_K,
                    hyper_param.auxiliary_size, 
                    0.5);
  model_ffm.SerializeToTXT("test_txt.ffm");
}

TEST(MODEL_TEST, BestModel) {
  // Init model
  HyperParam hyper_param = Init();
  Model model_ffm;
  model_ffm.Initialize(hyper_param.score_func,
                    hyper_param.loss_func,
                    hyper_param.num_feature,
                    hyper_param.num_field,
                    hyper_param.num_K, 2);
  real_t* w = model_ffm.GetParameter_w();
  real_t* v = model_ffm.GetParameter_v();
  real_t* b = model_ffm.GetParameter_b();
  // Set best model
  for (index_t i = 0; i < model_ffm.GetNumParameter_w(); ++i) {
    w[i] = 1;
  }
  for (index_t i = 0; i < model_ffm.GetNumParameter_v(); ++i) {
    v[i] = 2;
  }
  b[0] = 3;
  b[1] = 3;
  model_ffm.SetBestModel();
  // Shrink bacj
  for (index_t i = 0; i < model_ffm.GetNumParameter_w(); ++i) {
    w[i] = 0;
  }
  for (index_t i = 0; i < model_ffm.GetNumParameter_v(); ++i) {
    v[i] = 0;
  }
  b[0] = 0;
  b[1] = 0;
  model_ffm.Shrink();
  // Test
  for (index_t i = 0; i < model_ffm.GetNumParameter_w(); ++i) {
    EXPECT_FLOAT_EQ(w[i], 1);
  }
  for (index_t i = 0; i < model_ffm.GetNumParameter_v(); ++i) {
    EXPECT_FLOAT_EQ(v[i], 2);
  }
  EXPECT_FLOAT_EQ(b[0], 3);
  EXPECT_FLOAT_EQ(b[1], 3);
}

}   // namespace xLearn
