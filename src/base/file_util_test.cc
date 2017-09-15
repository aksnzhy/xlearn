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

TEST(FileTest, Serialize_and_Deserialize_string) {
  std::string filename = "/tmp/test.bin";
  FILE* file = OpenFileOrDie(filename.c_str(), "w");
  // Serialize
  std::string str_1("apple");
  std::string str_2("love");
  WriteStringToFile(file, str_1);
  WriteStringToFile(file, str_2);
  Close(file);
  str_1.clear();
  str_2.clear();
  // Deserialize
  file = OpenFileOrDie(filename.c_str(), "r");
  ReadStringFromFile(file, str_1);
  ReadStringFromFile(file, str_2);
  EXPECT_EQ(str_1, "apple");
  EXPECT_EQ(str_2, "love");
  Close(file);
  RemoveFile(filename.c_str());
}

TEST(FileTest, Serialize_and_Deserialize_file) {
   std::string filename = "/tmp/test.bin";
   FILE* file = OpenFileOrDie(filename.c_str(), "w");
   // Serialize
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

TEST(FileTest, HashFile) {
  FILE* file_1 = OpenFileOrDie("./tmp_1", "w");
  FILE* file_2 = OpenFileOrDie("./tmp_2", "w");
  FILE* file_3 = OpenFileOrDie("./tmp_3", "w");
  std::string str_1("affadada32sfsfsdse23");
  std::string str_2("jnfdj3278hjsldjksjd88ds");
  WriteStringToFile(file_1, str_1);
  WriteStringToFile(file_2, str_1);
  WriteStringToFile(file_3, str_2);
  Close(file_1);
  Close(file_2);
  Close(file_3);
  EXPECT_EQ(HashFile("./tmp_1"), HashFile("./tmp_2"));
  EXPECT_NE(HashFile("./tmp_2"), HashFile("./tmp_3"));
  RemoveFile("./tmp_1");
  RemoveFile("./tmp_2");
  RemoveFile("./tmp_3");
}

}  // namespace xLearn
