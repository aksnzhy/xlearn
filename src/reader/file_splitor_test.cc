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
This file tests the file_splitor.h file.
*/

#include "gtest/gtest.h"

#include <string>

#include "src/base/file_util.h"
#include "src/base/stringprintf.h"
#include "src/reader/file_splitor.h"

using std::string;

namespace xLearn {

const string kTestfilename = "/tmp/test_file";
const int kNumfolds = 5;
const int kNumOfLines = 1000 + 1;
const int kMaxLineSize = 100 * 1024; // 100 kb

class SpliterTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    FILE* file_ptr = OpenFileOrDie(kTestfilename.c_str(), "w");
    for (int i = 0; i < kNumOfLines; ++i) {
      string data = StringPrintf("%d\n", i);
      uint32 write_len =
        fwrite(data.c_str(), 1, data.size(), file_ptr);
      EXPECT_EQ(write_len, data.size());
    }
    Close(file_ptr);
    // split
    FileSpliter spliter;
    spliter.split(kTestfilename, kNumfolds);
  }
  virtual void TearDown() {
    for (int i = 0; i < kNumfolds; ++i) {
      string filename = StringPrintf("%s_%d", kTestfilename.c_str(), i);
      RemoveFile(filename.c_str());
    }
  }
};

TEST_F(SpliterTest, ReadTest) {
  char* line = new char[kMaxLineSize];
  // For each small file
  int num = -1;
  for (int i = 0; i < kNumfolds; ++i) {
    string filename = StringPrintf("%s_%d", kTestfilename.c_str(), i);
    FILE* file_ptr = OpenFileOrDie(filename.c_str(), "r");
    // Read every lines
    while (fgets(line, kMaxLineSize, file_ptr) != nullptr) {
      num++;
      string strTmp;
      int read_len = strlen(line);
      line[read_len - 1] = '\0';
      strTmp.assign(line);
      EXPECT_EQ(strTmp, StringPrintf("%d", num));
    }
    Close(file_ptr);
  }
}

} // namespace xLearn
