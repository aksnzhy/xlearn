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

This file tests the Reader class.
*/

#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "src/reader/reader.h"
#include "src/base/file_util.h"

using std::vector;
using std::string;

namespace xLearn {

const string kTestfilename = "./test_reader";
const string kStr = "0 1:0.123 1:0.123 1:0.123\n";
const string kStrFFM = "1 1:1:0.123 1:1:0.123 1:1:0.123\n";
const index_t kFeatureNum = 3;
const index_t kNumLines = 100000;
const index_t kNumSamples = 200;
const int iteration_num = 2000;

void WriteFile() {
  string lr_file = kTestfilename + "_LR.txt";
  string ffm_file = kTestfilename + "_ffm.txt";
  // Create libsvm file
  FILE* file_lr = OpenFileOrDie(lr_file.c_str(), "w");
  for (index_t i = 0; i < kNumLines; ++i) {
    uint32 write_len = fwrite(kStr.c_str(), 1, kStr.size(), file_lr);
    EXPECT_EQ(write_len, kStr.size());
  }
  Close(file_lr);
  // Create libffm file
  FILE* file_ffm = OpenFileOrDie(ffm_file.c_str(), "w");
  for (index_t i = 0; i < kNumLines; ++i) {
    uint32 write_len = fwrite(kStrFFM.c_str(), 1, kStrFFM.size(), file_ffm);
    EXPECT_EQ(write_len, kStrFFM.size());

  }
  Close(file_ffm);
}

void CheckLR(const DMatrix* matrix) {
  EXPECT_EQ(matrix->row_length, kNumSamples);
  // check the first element
  EXPECT_EQ(matrix->Y[0], (real_t)0);
  for (int i = 0; i < matrix->row_length; ++i) {
    SparseRow *row = matrix->row[i];
    int n = 0;
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      if (n == 0) {
        EXPECT_EQ(iter->field_id, 0);
        EXPECT_EQ(iter->feat_id, 0);
        EXPECT_FLOAT_EQ(iter->feat_val, 1.0);
      } else {
        EXPECT_EQ(iter->field_id, 0);
        EXPECT_EQ(iter->feat_id, 1);
        EXPECT_FLOAT_EQ(iter->feat_val, 0.123);
      }
      n++;
    }
    EXPECT_EQ(n, kFeatureNum+1);
  }

}

void CheckFFM(const DMatrix* matrix) {
  EXPECT_EQ(matrix->row_length, kNumSamples);
  // check the first element
  EXPECT_EQ(matrix->Y[0], (real_t)1);
  for (int i = 0; i < matrix->row_length; ++i) {
    SparseRow *row = matrix->row[i];
    int n = 0;
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      if (n == 0) {
        EXPECT_EQ(iter->field_id, 0);
        EXPECT_EQ(iter->feat_id, 0);
        EXPECT_FLOAT_EQ(iter->feat_val, 1.0);
      } else {
        EXPECT_EQ(iter->field_id, 1);
        EXPECT_EQ(iter->feat_id, 1);
        EXPECT_FLOAT_EQ(iter->feat_val, 0.123);
      }
      n++;
    }
    EXPECT_EQ(n, kFeatureNum+1);
  }
}

TEST(ReaderTest, SampleFromMemory) {
  WriteFile();
  string lr_file = kTestfilename + "_LR.txt";
  string ffm_file = kTestfilename + "_ffm.txt";
  // libsvm
  InmemReader reader_lr;
  reader_lr.Initialize(lr_file, kNumSamples);
  DMatrix* matrix = NULL;
  int i = 0;
  for (; i < iteration_num; ++i) {
    int record_num = reader_lr.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader_lr.Reset();
      continue;
    }
    EXPECT_EQ(record_num, kNumSamples);
    CheckLR(matrix);
  }
  EXPECT_EQ(i, iteration_num);
  // libffm
  matrix = nullptr;
  InmemReader reader_ffm;
  reader_ffm.Initialize(ffm_file, kNumSamples);
  for (i = 0; i < iteration_num; ++i) {
    int record_num = reader_ffm.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader_ffm.Reset();
      continue;
    }
    EXPECT_EQ(record_num, kNumSamples);
    CheckFFM(matrix);
  }
  EXPECT_EQ(i, iteration_num);
}

TEST(ReaderTest, ReadFromBinary) {
  string lr_file = kTestfilename + "_LR.txt";
  string ffm_file = kTestfilename + "_ffm.txt";
  // libsvm
  InmemReader reader_lr;
  reader_lr.Initialize(lr_file, kNumSamples);
  DMatrix* matrix = NULL;
  int i = 0;
  for (; i < iteration_num; ++i) {
    int record_num = reader_lr.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader_lr.Reset();
      continue;
    }
    EXPECT_EQ(record_num, kNumSamples);
    CheckLR(matrix);
  }
  EXPECT_EQ(i, iteration_num);
  // libffm
  matrix = nullptr;
  InmemReader reader_ffm;
  reader_ffm.Initialize(ffm_file, kNumSamples);
  for (i = 0; i < iteration_num; ++i) {
    int record_num = reader_ffm.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader_ffm.Reset();
      continue;
    }
    EXPECT_EQ(record_num, kNumSamples);
    CheckFFM(matrix);
  }
  EXPECT_EQ(i, iteration_num);
  string lr_bin = lr_file + ".bin";
  string ffm_bin = ffm_file + ".bin";
  RemoveFile(lr_file.c_str());
  RemoveFile(ffm_file.c_str());
  RemoveFile(lr_bin.c_str());
  RemoveFile(ffm_bin.c_str());
}

/*
TEST_F(ReaderTest, SampleFromDisk) {
  string lr_file = kTestfilename + "_LR.txt";
  string ffm_file = kTestfilename + "_ffm.txt";
  string csv_file = kTestfilename + "_csv.txt";
  // libsvm
  OndiskReader reader_lr;
  reader_lr.Initialize(lr_file, kNumSamples, parser_lr);
  DMatrix* matrix = nullptr;
  int i = 0;
  for (; i < iteration_num; ++i) {
    int record_num = reader_lr.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader_lr.Reset();
      continue;
    }
    EXPECT_EQ(record_num, kNumSamples);
    CheckLR(matrix);
  }
  EXPECT_EQ(i, iteration_num);
  // libffm
  OndiskReader reader_ffm;
  reader_ffm.Initialize(ffm_file, kNumSamples, parser_ffm);
  for (i = 0; i < iteration_num; ++i) {
    int record_num = reader_ffm.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader_ffm.Reset();
      continue;
    }
    EXPECT_EQ(record_num, kNumSamples);
    CheckFFM(matrix);
  }
  EXPECT_EQ(i, iteration_num);
  // csv
  OndiskReader reader_csv;
  reader_csv.Initialize(csv_file, kNumSamples, parser_csv);
  for (i = 0; i < iteration_num; ++i) {
    int record_num = reader_csv.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader_csv.Reset();
      continue;
    }
    EXPECT_EQ(record_num, kNumSamples);
    CheckLR(matrix);
  }
  EXPECT_EQ(i, iteration_num);
}
*/

Reader* CreateReader(const char* format_name) {
  return CREATE_READER(format_name);
}

TEST(READER_TEST, CreateReader) {
  EXPECT_TRUE(CreateReader("memory") != NULL);
  EXPECT_TRUE(CreateReader("disk") != NULL);
  EXPECT_TRUE(CreateReader("") == NULL);
  EXPECT_TRUE(CreateReader("unknow_name") == NULL);
}

} // namespace xLearn
