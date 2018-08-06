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
This file tests c_api.h file.
*/

#include "gtest/gtest.h"

#include "src/c_api/c_api.h"

TEST(C_API_TEST, Initialize) {
  XL xlearn;
  // Set
  EXPECT_EQ(XLearnHello(), 0);
  EXPECT_EQ(XLearnCreate("fm", &xlearn), 0);
  EXPECT_EQ(XLearnSetTrain(&xlearn, "./data_train.txt"), 0);
  EXPECT_EQ(XLearnSetTest(&xlearn, "./data_test.txt"), 0);
  EXPECT_EQ(XLearnSetValidate(&xlearn, "./data_validate.txt"), 0);
  EXPECT_EQ(XLearnSetTXTModel(&xlearn, "./txt_model"), 0);
  EXPECT_EQ(XLearnSetStr(&xlearn, "loss", "squared"), 0);
  EXPECT_EQ(XLearnSetStr(&xlearn, "metric", "auc"), 0);
  EXPECT_EQ(XLearnSetStr(&xlearn, "opt", "ftrl"), 0);
  EXPECT_EQ(XLearnSetStr(&xlearn, "log", "./log"), 0);
  EXPECT_EQ(XLearnSetInt(&xlearn, "k", 8), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "lr", 0.1), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "lambda", 0.1), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "alpha", 0.1), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "beta", 0.1), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "lambda_1", 0.1), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "lambda_2", 0.1), 0);
  EXPECT_EQ(XLearnSetFloat(&xlearn, "init", 0.1), 0);
  EXPECT_EQ(XLearnSetInt(&xlearn, "epoch", 3), 0);
  EXPECT_EQ(XLearnSetInt(&xlearn, "fold", 10), 0);
  EXPECT_EQ(XLearnSetInt(&xlearn, "nthread", 3), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "on_disk", true), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "lock_free", false), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "early_stop", false), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "quiet", true), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "norm", false), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "sign", true), 0);
  EXPECT_EQ(XLearnSetBool(&xlearn, "sigmoid", true), 0);
  EXPECT_EQ(XLearnSetInt(&xlearn, "block_size", 256), 0);
  EXPECT_EQ(XLearnShow(&xlearn), 0);
  // Get
  XLearn* xl = reinterpret_cast<XLearn*>(xlearn);
  EXPECT_EQ(xl->GetHyperParam().score_func, "fm");
  EXPECT_EQ(xl->GetHyperParam().train_set_file, "./data_train.txt");
  EXPECT_EQ(xl->GetHyperParam().test_set_file, "./data_test.txt");
  EXPECT_EQ(xl->GetHyperParam().validate_set_file, "./data_validate.txt");
  EXPECT_EQ(xl->GetHyperParam().txt_model_file, "./txt_model");
  EXPECT_EQ(xl->GetHyperParam().loss_func, "squared");
  EXPECT_EQ(xl->GetHyperParam().metric, "auc");
  EXPECT_EQ(xl->GetHyperParam().opt_type, "ftrl");
  EXPECT_EQ(xl->GetHyperParam().log_file, "./log");
  EXPECT_EQ(xl->GetHyperParam().num_K, 8);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().learning_rate, 0.1);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().regu_lambda, 0.1);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().model_scale, 0.1);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().alpha, 0.1);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().beta, 0.1);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().lambda_1, 0.1);
  EXPECT_FLOAT_EQ(xl->GetHyperParam().lambda_2, 0.1);
  EXPECT_EQ(xl->GetHyperParam().num_epoch, 3);
  EXPECT_EQ(xl->GetHyperParam().num_folds, 10);
  EXPECT_EQ(xl->GetHyperParam().thread_number, 3);
  EXPECT_EQ(xl->GetHyperParam().on_disk, true);
  EXPECT_EQ(xl->GetHyperParam().lock_free, false);
  EXPECT_EQ(xl->GetHyperParam().early_stop, false);
  EXPECT_EQ(xl->GetHyperParam().quiet, true);
  EXPECT_EQ(xl->GetHyperParam().norm, false);
  EXPECT_EQ(xl->GetHyperParam().sign, true);
  EXPECT_EQ(xl->GetHyperParam().sigmoid, true);
  EXPECT_EQ(xl->GetHyperParam().block_size, 256);
  EXPECT_EQ(XLearnHandleFree(&xlearn), 0);
}