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
This file tests the KVStore class file
*/

#include "gtest/gtest.h"

#include "src/distributed/parameter_server.h"

namespace xLearn {

TEST(KVStoreTest, GetServerId) {
  KVStore store;
  store.Initialize(3);
  EXPECT_EQ(store.GetServerId((index_t)0), (size_t)0);
  EXPECT_EQ(store.GetServerId((index_t)1), (size_t)1);
  EXPECT_EQ(store.GetServerId((index_t)2), (size_t)2);
  EXPECT_EQ(store.GetServerId((index_t)3), (size_t)0);
  EXPECT_EQ(store.GetServerId((index_t)4), (size_t)1);
  EXPECT_EQ(store.GetServerId((index_t)5), (size_t)2);
  EXPECT_EQ(store.GetServerId((index_t)6), (size_t)0);
  EXPECT_EQ(store.GetServerId((index_t)7), (size_t)1);
  EXPECT_EQ(store.GetServerId((index_t)8), (size_t)2);
  EXPECT_EQ(store.GetServerId((index_t)9), (size_t)0);
}

TEST(KVStoreTest, FeatMap) {
  KVStore store;
  store.Initialize(3);
  EXPECT_EQ(store.FeatMap((index_t)0), (index_t)0);
  EXPECT_EQ(store.FeatMap((index_t)1), (index_t)0);
  EXPECT_EQ(store.FeatMap((index_t)2), (index_t)0);
  EXPECT_EQ(store.FeatMap((index_t)3), (index_t)1);
  EXPECT_EQ(store.FeatMap((index_t)4), (index_t)1);
  EXPECT_EQ(store.FeatMap((index_t)5), (index_t)1);
  EXPECT_EQ(store.FeatMap((index_t)6), (index_t)2);
  EXPECT_EQ(store.FeatMap((index_t)7), (index_t)2);
  EXPECT_EQ(store.FeatMap((index_t)8), (index_t)2);
  EXPECT_EQ(store.FeatMap((index_t)9), (index_t)3);
}

}  // namespace xLearn