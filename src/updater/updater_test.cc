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

This file tests a set of updaters.
*/

#include "gtest/gtest.h"

#include "src/base/common.h"

#include "src/update/updater.h"
#include "src/update/adadelta_updater.h"
#include "src/update/adagrad_updater.h"
#include "src/update/adam_updater.h"
#include "src/update/momentum_updater.h"
#include "src/update/rmsprop_updater.h"

namespace f2m {

Updater* CreateUpdater(const char* format_name) {
  return CREATE_UPDATER(format_name);
}

TEST(UPDATER_TEST, CreateUpdater) {
  EXPECT_TRUE(CreateUpdater("sgd") != NULL);
  EXPECT_TRUE(CreateUpdater("adadelta") != NULL);
  EXPECT_TRUE(CreateUpdater("adagrad") != NULL);
  EXPECT_TRUE(CreateUpdater("adam") != NULL);
  EXPECT_TRUE(CreateUpdater("momentum") != NULL);
  EXPECT_TRUE(CreateUpdater("rmsprop") != NULL);
  EXPECT_TRUE(CreateUpdater("Unknow_Updater") == NULL);
}

} // namespace f2m
