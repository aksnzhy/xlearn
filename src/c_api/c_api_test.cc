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

This file tests c_api.h
*/

#include "gtest/gtest.h"

#include "src/c_api/c_api.h"

TEST(C_API_TEST, Initialize) {
  XL xlearn;
  // Set
  EXPECT_EQ(XLearnHello(), 0);
  EXPECT_EQ(XLearnCreate("linear", &xlearn), 0);
  EXPECT_EQ(XLearnSetTrain(&xlearn, "./data_train.txt"), 0);
  EXPECT_EQ(XLearnSetTest(&xlearn, "./data_test.txt"), 0);
  EXPECT_EQ(XLearnSetValidate(&xlearn, "./data_validate.txt"), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "on_disk", true), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "quiet", true), 0);
  EXPECT_EQ(XLearnSetStr(&xlearn, "loss", "squared"), 0);
  EXPECT_EQ(XLearnSetStr(&xlearn, "metric", "auc"), 0);
  EXPECT_EQ(XLearnSetInt(&xlearn, "block_size", 256), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "lr", 0.1), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "lambda", 0.05), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "init", 0.22), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "alpha", 1.0), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "beta", 1.0), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "lambda_1", 1.0), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "lambda_2", 1.0), 0);
  EXPECT_EQ(XLearnSetInt(&xlearn, "epoch", 5), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "norm", false), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "lock_free", true), 0);
  EXPECT_EQ(XLearnSetInt(&xlearn, "k", 8), 0);
  EXPECT_EQ(XLearnSetStr(&xlearn, "log", "./log.txt"), 0);
  EXPECT_EQ(XLearnSetInt(&xlearn, "fold", 10), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "early_stop", false), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "sign", true), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "sigmoid", true), 0);
  EXPECT_EQ(XLearnShow(&xlearn), 0);
  // Test
  XLearn* xl = reinterpret_cast<XLearn*>(xlearn);
  EXPECT_EQ(xl->GetHyperParam().score_func, "linear");
  EXPECT_EQ(xl->GetHyperParam().train_set_file, "./data_train.txt");
  EXPECT_EQ(xl->GetHyperParam().test_set_file, "./data_test.txt");
  EXPECT_EQ(xl->GetHyperParam().validate_set_file, "./data_validate.txt");
  EXPECT_EQ(xl->GetHyperParam().on_disk, true);
  EXPECT_EQ(xl->GetHyperParam().quiet, true);
  EXPECT_EQ(xl->GetHyperParam().loss_func, "squared");
  EXPECT_EQ(xl->GetHyperParam().metric, "auc");
  EXPECT_EQ(xl->GetHyperParam().block_size, 256);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().learning_rate, 0.1);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().regu_lambda, 0.05);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().model_scale, 0.22);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().alpha, 1);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().beta, 1);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().lambda_1, 1);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().lambda_2, 1);
  EXPECT_EQ(xl->GetHyperParam().num_epoch, 5);
  EXPECT_EQ(xl->GetHyperParam().norm, false);
  EXPECT_EQ(xl->GetHyperParam().lock_free, true);
  EXPECT_EQ(xl->GetHyperParam().num_K, 8);
  EXPECT_EQ(xl->GetHyperParam().log_file, "./log.txt");
  EXPECT_EQ(xl->GetHyperParam().num_folds, 10);
  EXPECT_EQ(xl->GetHyperParam().early_stop, false);
  EXPECT_EQ(xl->GetHyperParam().sign, true);
  EXPECT_EQ(xl->GetHyperParam().sigmoid, true);
}