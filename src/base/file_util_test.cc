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

This file tests file_util.h
*/

#include "gtest/gtest.h"

#include <vector>
#include <string>

#include "src/base/file_util.h"

namespace xLearn {

TEST(FileTest, Serialize_and_Deserialize_buffer) {
  std::vector<int> array;
  for (int i = 0; i < 5; ++i) {
    array.push_back(i);
  }
  char* buffer = nullptr;
  serialize_vector(array, buffer);
  size_t len = sizeof(size_t) + sizeof(int) * array.size();
  array.clear();
  deserialize_vector(buffer, len, array);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(array[i], i);
  }
}

TEST(FileTest, Serialize_and_Deserialize_file) {
   std::string filename = "/tmp/test.bin";
   // Serialize
   FILE* file = OpenFileOrDie(filename.c_str(), "w");
   std::vector<int> array;
   for (int i = 0; i < 10; ++i) {
     array.push_back(i);
   }
   WriteVectorToFile(file, array);
   array.clear();
   for (int i = 0; i < 12; ++i) {
     array.push_back(i);
   }
   WriteVectorToFile(file, array);
   array.clear();
   for (int i = 0; i < 15; ++i) {
     array.push_back(i);
   }
   WriteVectorToFile(file, array);
   array.clear();
   Close(file);
   // Deserialize
   file = OpenFileOrDie(filename.c_str(), "r");
   ReadVectorFromFile(file, array);
   for (int i = 0; i < 10; ++i) {
     EXPECT_EQ(array[i], i);
   }
   array.clear();
   std::vector<int> array_1;
   ReadVectorFromFile(file, array_1);
   for (int i = 0; i < 12; ++i) {
     EXPECT_EQ(array[i], i);
   }
   array.clear();
   ReadVectorFromFile(file, array);
   for (int i = 0; i < 15; ++i) {
     EXPECT_EQ(array[i], i);
   }
   Close(file);
   RemoveFile(filename.c_str());
}

}  // namespace xLearn
