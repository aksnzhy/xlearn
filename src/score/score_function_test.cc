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
This file tests the Score class.
*/

#include "gtest/gtest.h"

#include "src/score/score_function.h"

namespace xLearn {

Score* CreateScore(const char* format_name) {
  return CREATE_SCORE(format_name);
}

TEST(SCORE_TEST, Create_Score) {
  EXPECT_TRUE(CreateScore("linear") != NULL);
  EXPECT_TRUE(CreateScore("fm") != NULL);
  EXPECT_TRUE(CreateScore("ffm") != NULL);
  EXPECT_TRUE(CreateScore("") == NULL);
  EXPECT_TRUE(CreateScore("unknow_name") == NULL);
}

}  // namespace xLearn
