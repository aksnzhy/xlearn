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

index_t K = 24;
index_t kFeature_num = 3;
index_t kField_num = 3;
index_t kLine = 10;

TEST(LOSS, Predict_Linear) {
  // Create Model for Linear
  index_t parameter_num = kFeature_num + 1;
  Model model_lr(parameter_num);
  std::vector<real_t>* para = model_lr.GetParameter();
  for (size_t i = 0; i < para->size(); ++i) {
    (*para)[i] = 2.0;
  }
  // Create Data matrix
  DMatrix matrix(kLine);
  matrix.InitSparseRow();
  for (int i = 0; i < kLine; ++i) {
    matrix.Y[i] = 0;
    matrix.row[i]->Resize(kFeature_num + 1);
    for (int j = 0; j < kFeature_num+1; ++j) {
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
    EXPECT_FLOAT_EQ(pred[i], 8.0);
  }
}

TEST(LOSS, Predict_FM) {
  // Create Model for FM
  index_t parameter_num = kFeature_num + 1 + kFeature_num*K;
  Model model_lr(parameter_num);
  std::vector<real_t>* para = model_lr.GetParameter();
  for (size_t i = 0; i < para->size(); ++i) {
    (*para)[i] = 2.0;
  }
  // Create Data matrix
  DMatrix matrix(kLine);
  matrix.InitSparseRow();
  for (int i = 0; i < kLine; ++i) {
    matrix.Y[i] = 0;
    matrix.row[i]->Resize(kFeature_num + 1);
    for (int j = 0; j < kFeature_num+1; ++j) {
      matrix.row[i]->X[j] = 1.0;
      matrix.row[i]->idx[j] = j;
    }
  }
  // Create HyperParam
  HyperParam hyper_param;
  hyper_param.num_feature = kFeature_num;
  hyper_param.num_K = K;
  // Create score function
  FMScore score;
  score.Initialize(hyper_param);
  // Create loss function
  TestLoss loss;
  loss.Initialize(&score);
  // Test
  std::vector<real_t> pred(kLine);
  loss.Predict(&matrix, &model_lr, pred);
  for (int i = 0; i < kLine; ++i) {
    EXPECT_FLOAT_EQ(pred[i], 296.0);
  }
}

TEST(LOSS, Predict_FFM) {
  // Create Model for FM
  index_t parameter_num = kFeature_num + 1 + kFeature_num*K*kField_num;
  Model model_lr(parameter_num);
  std::vector<real_t>* para = model_lr.GetParameter();
  for (size_t i = 0; i < para->size(); ++i) {
    (*para)[i] = 2.0;
  }
  // Create Data matrix
  DMatrix matrix(kLine);
  matrix.InitSparseRow(true);
  for (int i = 0; i < kLine; ++i) {
    matrix.Y[i] = 0;
    matrix.row[i]->Resize(kFeature_num + 1);
    for (int j = 0; j < kFeature_num+1; ++j) {
      matrix.row[i]->X[j] = 1.0;
      matrix.row[i]->idx[j] = j;
      matrix.row[i]->field[j] = j;
    }
  }
  // Create HyperParam
  HyperParam hyper_param;
  hyper_param.num_feature = kFeature_num;
  hyper_param.num_K = K;
  hyper_param.num_field = kField_num;
  // Create score function
  FFMScore score;
  score.Initialize(hyper_param);
  // Create loss function
  TestLoss loss;
  loss.Initialize(&score);
  // Test
  std::vector<real_t> pred(kLine);
  loss.Predict(&matrix, &model_lr, pred);
  for (int i = 0; i < kLine; ++i) {
    EXPECT_FLOAT_EQ(pred[i], 296.0);
  }
}

Loss* CreateLoss(const char* format_name) {
  return CREATE_LOSS(format_name);
}

TEST(LOSS_TEST, Create_Loss) {
  EXPECT_TRUE(CreateLoss("squared") != NULL);
  EXPECT_TRUE(CreateLoss("hinge") != NULL);
  EXPECT_TRUE(CreateLoss("cross-entropy") != NULL);
  EXPECT_TRUE(CreateLoss("abs") != NULL);
  EXPECT_TRUE(CreateLoss("") == NULL);
  EXPECT_TRUE(CreateLoss("unknow_name") == NULL);
}

} // namespace xLearn
