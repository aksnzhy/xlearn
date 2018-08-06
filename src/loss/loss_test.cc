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
This file tests the Loss class.
*/

#include "gtest/gtest.h"

#include <vector>

#include "src/loss/loss.h"
#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/data/model_parameters.h"
#include "src/data/hyper_parameters.h"
#include "src/score/linear_score.h"
#include "src/score/fm_score.h"
#include "src/score/ffm_score.h"

namespace xLearn {

class TestLoss : public Loss {
 public:
  TestLoss() { }
  ~TestLoss() { }

  void Evalute(const std::vector<real_t>& pred,
               const std::vector<real_t>& label) { return; }

  void CalcGrad(const DMatrix* data_matrix,
                Model& model) { return; }

  void CalcGradDist(DMatrix* data_matrix,
                    Model& model,
                    std::vector<real_t>& grad) { return; }

  std::string loss_type() { return "test"; }

 private:
  DISALLOW_COPY_AND_ASSIGN(TestLoss);
};

HyperParam param;
const index_t kLine = 10;

class LossTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.loss_func = "sqaured";
    param.score_func = "linear";
    param.num_feature = 3;
    param.num_field = 3;
    param.num_K = 24;
    param.auxiliary_size = 2;
  }
};

TEST_F(LossTest, Predict_Linear) {
  // Create Model for Linear Re
  Model model_lr;
  model_lr.Initialize(param.score_func,
                  param.loss_func,
                  param.num_feature,
                  param.num_field,
                  param.num_K,
                  param.auxiliary_size);
  real_t* w = model_lr.GetParameter_w();
  index_t num_w = model_lr.GetNumParameter_w();
  for (size_t i = 0; i < num_w; ++i) {
    w[i] = 2.0;
  }
  // Create Data matrix
  DMatrix matrix;
  matrix.ResetMatrix(kLine);
  for (int i = 0; i < kLine; ++i) {
    matrix.Y[i] = 0;
    matrix.row[i] = new SparseRow;
    for (int j = 0; j < param.num_feature; ++j) {
      matrix.AddNode(i, j, 1.0);
    }
  }
  // Create loss function
  TestLoss loss;
  Score* score = new LinearScore;
  size_t threadNumber = std::thread::hardware_concurrency();
  ThreadPool* pool = new ThreadPool(threadNumber);
  loss.Initialize(score, pool);
  // Test
  std::vector<real_t> pred(kLine);
  loss.Predict(&matrix, model_lr, pred);
  for (int i = 0; i < kLine; ++i) {
    EXPECT_FLOAT_EQ(pred[i], 6.0);
  }
}

TEST_F(LossTest, Predict_FM) {
  // Create Model for FM
  Model model_fm;
  param.score_func = "fm";
  model_fm.Initialize(param.score_func,
                  param.loss_func,
                  param.num_feature,
                  param.num_field,
                  param.num_K,
                  param.auxiliary_size);
  real_t* w = model_fm.GetParameter_w();
  index_t num_w = model_fm.GetNumParameter_w();
  for (size_t i = 0; i < num_w; ++i) {
    w[i] = 2.0;
  }
  real_t* v = model_fm.GetParameter_v();
  index_t num_v = model_fm.GetNumParameter_v();
  for (size_t i = 0; i < num_v; ++i) {
    v[i] = 1.0;
  }
  // Create Data matrix
  DMatrix matrix;
  matrix.ResetMatrix(kLine);
  for (int i = 0; i < kLine; ++i) {
    matrix.Y[i] = 0;
    matrix.row[i] = new SparseRow;
    for (int j = 0; j < param.num_feature; ++j) {
      matrix.AddNode(i, j, 1.0);
    }
  }
  // Create loss function
  TestLoss loss;
  Score* score = new FMScore;
  size_t threadNumber = std::thread::hardware_concurrency();
  ThreadPool* pool = new ThreadPool(threadNumber);
  loss.Initialize(score, pool);
  // Test
  std::vector<real_t> pred(kLine);
  loss.Predict(&matrix, model_fm, pred);
  for (int i = 0; i < kLine; ++i) {
    // 6 + 24*1*3
    EXPECT_FLOAT_EQ(pred[i], 78);
  }
}

TEST_F(LossTest, Predict_FFM) {
  // Create Model for FFM
  Model model_ffm;
  param.score_func = "ffm";
  model_ffm.Initialize(param.score_func,
                   param.loss_func,
                   param.num_feature,
                   param.num_field,
                   param.num_K,
                   param.auxiliary_size);
  real_t* w = model_ffm.GetParameter_w();
  index_t num_w = model_ffm.GetNumParameter_w();
  for (size_t i = 0; i < num_w; ++i) {
    w[i] = 2.0;
  }
  real_t* v = model_ffm.GetParameter_v();
  index_t num_v = model_ffm.GetNumParameter_v();
  for (size_t i = 0; i < num_v; ++i) {
    v[i] = 1.0;
  }
  // Create Data matrix
  DMatrix matrix;
  matrix.ResetMatrix(kLine);
  for (int i = 0; i < kLine; ++i) {
    matrix.Y[i] = 0;
    matrix.row[i] = new SparseRow;
    for (int j = 0; j < param.num_feature; ++j) {
      matrix.AddNode(i, j, 1.0, j);
    }
  }
  // Create loss function
  TestLoss loss;
  Score* score = new FFMScore;
  size_t threadNumber = std::thread::hardware_concurrency();
  ThreadPool* pool = new ThreadPool(threadNumber);
  loss.Initialize(score, pool);
  // Test
  std::vector<real_t> pred(kLine);
  loss.Predict(&matrix, model_ffm, pred);
  for (int i = 0; i < kLine; ++i) {
    // 6 + 24*1*3
    EXPECT_FLOAT_EQ(pred[i], 78);
  }
}

Loss* CreateLoss(const char* format_name) {
  return CREATE_LOSS(format_name);
}

TEST_F(LossTest, Create_Loss) {
  EXPECT_TRUE(CreateLoss("squared") != NULL);
  EXPECT_TRUE(CreateLoss("cross-entropy") != NULL);
  EXPECT_TRUE(CreateLoss("") == NULL);
  EXPECT_TRUE(CreateLoss("unknow_name") == NULL);
}

} // namespace xLearn
