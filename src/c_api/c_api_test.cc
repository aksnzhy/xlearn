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
  XLearnHandle xlearn;
  EXPECT_EQ(XLearnCreate("linear", &xlearn), 0);
  EXPECT_EQ(XLearnSetTrain(&xlearn, "./data_train.txt"), 0);
  EXPECT_EQ(XLearnSetTest(&xlearn, "./data_test.txt"), 0);
  EXPECT_EQ(XLearnSetValidate(&xlearn, "./data_validate.txt"), 0);
  EXPECT_EQ(XLearnFit(&xlearn, "./model.bin"), 0);
}