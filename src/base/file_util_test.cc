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
This file tests file_util.h file.
*/

#include "gtest/gtest.h"

#include <vector>
#include <string>

#include "src/base/file_util.h"

TEST(FileTest, File_Exist) {
  std::string filename = "/tmp/test";
  FILE* file = OpenFileOrDie(filename.c_str(), "w");
  bool bo = FileExist(filename.c_str());
  EXPECT_EQ(bo, true);
  bo = FileExist("/tmp/non-exist-file");
  EXPECT_EQ(bo, false);
  Close(file);
  RemoveFile(filename.c_str());
}

TEST(FileTest, Get_File_Size) {
  std::string filename = "/tmp/test";
  FILE* file_w = OpenFileOrDie(filename.c_str(), "w");
  int number = 999;
  WriteDataToDisk(file_w, (char*)&number, sizeof(number));
  uint64 size = GetFileSize(file_w);
  EXPECT_EQ(size, 4);
  Close(file_w);
  RemoveFile(filename.c_str());
}

TEST(FileTest, Get_One_Line) {
  std::string filename = "/tmp/test";
  FILE* file_w = OpenFileOrDie(filename.c_str(), "w");
  std::string w_str("apple\n");
  for (int i = 0; i < 3; ++i) {
    fwrite(w_str.c_str(), 1, w_str.size(), file_w);
  }
  Close(file_w);
  FILE* file_r = OpenFileOrDie(filename.c_str(), "r");
  for (int i = 0; i < 3; ++i) {
    std::string r_str;
    GetLine(file_r, r_str);
    EXPECT_EQ(r_str, "apple");
  }
  Close(file_r);
  RemoveFile(filename.c_str());
}

TEST(FileTest, Write_and_Read_Data) {
  std::string filename = "/tmp/test";
  FILE* file_w = OpenFileOrDie(filename.c_str(), "w");
  int number = 999;
  WriteDataToDisk(file_w, (char*)&number, sizeof(number));
  Close(file_w);
  int read = 0;
  FILE* file_r = OpenFileOrDie(filename.c_str(), "r");
  ReadDataFromDisk(file_r, (char*)&read, sizeof(read));
  EXPECT_EQ(read, number);
  Close(file_r);
  RemoveFile(filename.c_str());
}

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

TEST(FileTest, Serialize_and_Deserialize_vector) {
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
   ReadVectorFromFile(file, array);
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

TEST(FileTest, ReadFile) {
  FILE* file = OpenFileOrDie("./tmp.bin", "w");
  int num = 999;
  WriteDataToDisk(file, (char*)&num, sizeof(num));
  Close(file);
  char* ch_num = nullptr;
  uint64 len = ReadFileToMemory("./tmp.bin", &ch_num);
  EXPECT_EQ(len, sizeof(num));
  EXPECT_EQ((*(int*)ch_num), 999);
  RemoveFile("./tmp.bin");
}
