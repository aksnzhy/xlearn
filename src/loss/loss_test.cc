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

  real_t Evalute(const std::vector<real_t>& pred,
                 const std::vector<real_t>& label) { return 0.0; }

  void CalcGrad(const DMatrix* data_matrix,
                Model* model,
                Updater* updater) { }

 private:
  DISALLOW_COPY_AND_ASSIGN(TestLoss);
};

const index_t kLine = 10;

HyperParam param;
class LossTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    param.learning_rate = 0.1;
    param.regu_lambda = 0;
    param.decay_rate_1 = 0.91;
    param.num_param = 3;
    param.loss_func = "sqaured";
    param.score_func = "linear";
    param.num_feature = 3;
    param.num_field = 3;
    param.num_K = 24;
  }
};

TEST_F(LossTest, Predict_Linear) {
  // Create Model for Linear
  Model model_lr;
  model_lr.Initialize(param.num_param,
                  param.score_func,
                  param.loss_func,
                  param.num_feature,
                  param.num_field,
                  param.num_K,
                  false);
  std::vector<real_t>* para = model_lr.GetParameter();
  for (size_t i = 0; i < para->size(); ++i) {
    (*para)[i] = 2.0;
  }
  // Create Data matrix
  DMatrix matrix(kLine);
  matrix.InitSparseRow();
  for (int i = 0; i < kLine; ++i) {
    matrix.Y[i] = 0;
    matrix.row[i]->Resize(param.num_feature);
    for (int j = 0; j < param.num_feature; ++j) {
      matrix.row[i]->X[j] = 1.0;
      matrix.row[i]->idx[j] = j;
    }
  }
  // Create score function
  LinearScore score;
  // Create loss function
  TestLoss loss;
  loss.Initialize(&score);
  // Test
  std::vector<real_t> pred(kLine);
  loss.Predict(&matrix, &model_lr, pred);
  for (int i = 0; i < kLine; ++i) {
    EXPECT_FLOAT_EQ(pred[i], 6.0);
  }
}

TEST_F(LossTest, Predict_FM) {
  // Create Model for FM
  param.num_param = 3 + 3*24;
  Model model_fm;
  model_fm.Initialize(param.num_param,
                  param.score_func,
                  param.loss_func,
                  param.num_feature,
                  param.num_field,
                  param.num_K,
                  false);
  std::vector<real_t>* para = model_fm.GetParameter();
  for (size_t i = 0; i < para->size(); ++i) {
    (*para)[i] = 2.0;
  }
  // Create Data matrix
  DMatrix matrix(kLine);
  matrix.InitSparseRow();
  for (int i = 0; i < kLine; ++i) {
    matrix.Y[i] = 0;
    matrix.row[i]->Resize(param.num_feature);
    for (int j = 0; j < param.num_feature; ++j) {
      matrix.row[i]->X[j] = 1.0;
      matrix.row[i]->idx[j] = j;
    }
  }
  // Create score function
  FMScore score;
  score.Initialize(param.num_feature,
                param.num_K,
                param.num_field);
  // Create loss function
  TestLoss loss;
  loss.Initialize(&score);
  // Test
  std::vector<real_t> pred(kLine);
  loss.Predict(&matrix, &model_fm, pred);
  for (int i = 0; i < kLine; ++i) {
    EXPECT_FLOAT_EQ(pred[i], 294.0);
  }
}

TEST_F(LossTest, Predict_FFM) {
  // Create Model for FFM
  index_t parameter_num = 3 + 3*3*24;
  param.num_param = parameter_num;
  Model model_ffm;
  model_ffm.Initialize(param.num_param,
                   param.score_func,
                   param.loss_func,
                   param.num_feature,
                   param.num_field,
                   param.num_K,
                   false);
  std::vector<real_t>* para = model_ffm.GetParameter();
  for (size_t i = 0; i < para->size(); ++i) {
    (*para)[i] = 2.0;
  }
  // Create Data matrix
  DMatrix matrix(kLine);
  matrix.InitSparseRow(true);
  for (int i = 0; i < kLine; ++i) {
    matrix.Y[i] = 0;
    matrix.row[i]->Resize(param.num_feature);
    for (int j = 0; j < param.num_feature; ++j) {
      matrix.row[i]->X[j] = 1.0;
      matrix.row[i]->idx[j] = j;
      matrix.row[i]->field[j] = j;
    }
  }
  // Create score function
  FFMScore score;
  score.Initialize(param.num_feature,
                 param.num_K,
                 param.num_field);
  // Create loss function
  TestLoss loss;
  loss.Initialize(&score);
  // Test
  std::vector<real_t> pred(kLine);
  loss.Predict(&matrix, &model_ffm, pred);
  for (int i = 0; i < kLine; ++i) {
    EXPECT_FLOAT_EQ(pred[i], 294.0);
  }
}

TEST_F(LossTest, Sigmoid_Test) {
  std::vector<real_t> pred(6);
  pred[0] = 0.5;
  pred[1] = 3;
  pred[2] = 20;
  pred[3] = -0.5;
  pred[4] = -3;
  pred[5] = -20;
  std::vector<real_t> new_pred(pred.size());
  // Create score function
  LinearScore score;
  // Create loss function
  TestLoss loss;
  loss.Initialize(&score);
  loss.Sigmoid(pred, new_pred);
  EXPECT_GT(new_pred[0], 0.5);
  EXPECT_GT(new_pred[1], 0.5);
  EXPECT_GT(new_pred[2], 0.5);
  EXPECT_LT(new_pred[3], 0.5);
  EXPECT_LT(new_pred[4], 0.5);
  EXPECT_LT(new_pred[5], 0.5);
}

Loss* CreateLoss(const char* format_name) {
  return CREATE_LOSS(format_name);
}

TEST_F(LossTest, Create_Loss) {
  EXPECT_TRUE(CreateLoss("squared") != NULL);
  EXPECT_TRUE(CreateLoss("hinge") != NULL);
  EXPECT_TRUE(CreateLoss("cross-entropy") != NULL);
  EXPECT_TRUE(CreateLoss("abs") != NULL);
  EXPECT_TRUE(CreateLoss("") == NULL);
  EXPECT_TRUE(CreateLoss("unknow_name") == NULL);
}

} // namespace xLearn
