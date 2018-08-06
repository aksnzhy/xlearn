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
This file tests levenshtein_distance.h file.
*/

#include "gtest/gtest.h"

#include <vector>
#include <string>
#include <iostream>

#include "src/base/common.h"
#include "src/base/levenshtein_distance.h"

typedef std::vector<std::string> StringList;

TEST(LDISTANCE_TEST, Find) {
  StringList list;
  list.push_back("alex");
  list.push_back("apple");
  list.push_back("mac");
  StrSimilar ss;
  EXPECT_EQ(ss.Find(std::string("alex"), list), true);
  EXPECT_EQ(ss.Find(std::string("apple"), list), true);
  EXPECT_EQ(ss.Find(std::string("mac"), list), true);
  EXPECT_EQ(ss.Find(std::string("zz"), list), false);
}

TEST(LDISTANCE_TEST, FindSimilar) {
  StringList list;
  list.push_back("alex");
  list.push_back("apple");
  list.push_back("mac");
  StrSimilar ss;
  std::string result;
  // Equal
  EXPECT_EQ(ss.FindSimilar(std::string("alex"), list, result), 0);
  EXPECT_EQ(result, std::string("alex"));
  EXPECT_EQ(ss.FindSimilar(std::string("apple"), list, result), 0);
  EXPECT_EQ(result, std::string("apple"));
  EXPECT_EQ(ss.FindSimilar(std::string("mac"), list, result), 0);
  EXPECT_EQ(result, std::string("mac"));
  // Not equal
  EXPECT_EQ(ss.FindSimilar(std::string("alexx"), list, result), 1);
  EXPECT_EQ(result, std::string("alex"));
  EXPECT_EQ(ss.FindSimilar(std::string("apzple"), list, result), 1);
  EXPECT_EQ(result, std::string("apple"));
  EXPECT_EQ(ss.FindSimilar(std::string("maccz"), list, result), 2);
  EXPECT_EQ(result, std::string("mac"));
}
