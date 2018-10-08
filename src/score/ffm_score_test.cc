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
This file tests the FFMScore class.
*/

#include "gtest/gtest.h"

#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"
#include "src/score/score_function.h"
#include "src/score/ffm_score.h"

namespace xLearn {

TEST(FFMScore_Test, calc_score) {
  for (index_t k = 1; k < 100; ++k) {
    // Init hyper_param
    HyperParam param;
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.loss_func = "squared";
    param.score_func = "ffm";
    param.num_feature = 3;
    param.num_K = k;
    param.num_field = 3;
    // Init SparseRow
    SparseRow row(param.num_feature);
    for (index_t i = 0; i < param.num_feature; ++i) {
      row[i].feat_id = i;
      row[i].feat_val = 2.0;
      row[i].field_id = i;
    }
    // Init model
    Model model;
    model.Initialize(param.score_func,
                param.loss_func,
                param.num_feature,
                param.num_field,
                param.num_K, 2);
    real_t* w = model.GetParameter_w();
    index_t num_w = model.GetNumParameter_w();
    for (index_t i = 0; i < num_w; ++i) {
      w[i] = 1.0;
    }
    real_t* v = model.GetParameter_v();
    index_t k_aligned = model.get_aligned_k();
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      for (index_t f = 0; f < model.GetNumField(); ++f) {
        for (index_t d = 0; d < k_aligned; ) {
          for (index_t s = 0; s < kAlign; s++, v++, d++) {
            v[0] = (d < model.GetNumK()) ? 1.0 : 0.0;
            v[kAlign] = 1.0;
          }
          v += kAlign;
        }
      }
    }
    model.GetParameter_b()[0] = 0.0;
    FFMScore score;
    for (size_t i = 0; i < 10; ++i) {
      real_t val = score.CalcScore(&row, model);
      // 6 + 24*4*3 = 294
      EXPECT_FLOAT_EQ(val, 6+k*4*3);
    }
  }
}

TEST(FFMScore_Test, calc_score_overflow) {
  for (index_t k = 1; k < 100; ++k) {
    // Init hyper_param
    HyperParam param;
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.loss_func = "squared";
    param.score_func = "ffm";
    param.num_feature = 3;
    param.num_K = k;
    param.num_field = 3;
    // Init SparseRow
    SparseRow row(param.num_feature*2);
    for (index_t i = 0; i < param.num_feature*2; ++i) {
      row[i].feat_id = i;
      row[i].feat_val = 2.0;
      row[i].field_id = i;
    }
    // Init model
    Model model;
    model.Initialize(param.score_func,
                param.loss_func,
                param.num_feature,
                param.num_field,
                param.num_K, 2);
    real_t* w = model.GetParameter_w();
    index_t num_w = model.GetNumParameter_w();
    for (index_t i = 0; i < num_w; ++i) {
      w[i] = 1.0;
    }
    real_t* v = model.GetParameter_v();
    index_t k_aligned = model.get_aligned_k();
    for (index_t j = 0; j < model.GetNumFeature(); ++j) {
      for (index_t f = 0; f < model.GetNumField(); ++f) {
        for (index_t d = 0; d < k_aligned; ) {
          for (index_t s = 0; s < kAlign; s++, v++, d++) {
            v[0] = (d < model.GetNumK()) ? 1.0 : 0.0;
            v[kAlign] = 1.0;
          }
          v += kAlign;
        }
      }
    }
    model.GetParameter_b()[0] = 0.0;
    FFMScore score;
    for (size_t i = 0; i < 10; ++i) {
      real_t val = score.CalcScore(&row, model);
      EXPECT_FLOAT_EQ(val, 6+k*4*3);
    }
  }
}

} // namespace xLearn
