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

This file tests reader.h
*/

#include "gtest/gtest.h"

#include <unistd.h> // for usleep()

#include <string>
#include <vector>
#include <chrono>  // for std::chrono

#include "src/base/file_util.h"
#include "src/reader/reader.h"
#include "src/data/data_structure.h"

using std::vector;
using std::string;

namespace f2m {

const string kTestfilename = "/tmp/test_reader";
const string kStr = "0 1:0.123 2:0.123 3:0.123\n";
const string kStrFFM = "1 1:1:0.123 2:2:0.123 3:3:0.123\n";
const index_t kFeatureNum = 3;
const index_t kNumLines = 100000;
const index_t kNumSamples = 10000;
const int iteration_num = 300;

Parser* parser_lr = new LibsvmParser;
Parser* parser_ffm = new FFMParser;

class ReaderTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    string lr_file = kTestfilename + "_LR.txt";
    string ffm_file = kTestfilename + "_ffm.txt";
    FILE* file_lr = OpenFileOrDie(lr_file.c_str(), "w");
    for (index_t i = 0; i < kNumLines; ++i) {
      uint32 write_len = fwrite(kStr.c_str(), 1, kStr.size(), file_lr);
      EXPECT_EQ(write_len, kStr.size());
    }
    Close(file_lr);
    FILE* file_ffm = OpenFileOrDie(ffm_file.c_str(), "w");
    for (index_t i = 0; i < kNumLines; ++i) {
      uint32 write_len = fwrite(kStrFFM.c_str(), 1, kStrFFM.size(), file_ffm);
      EXPECT_EQ(write_len, kStrFFM.size());

    }
    Close(file_ffm);
  }
  virtual void TearDown() {
    string lr_file = kTestfilename + "_LR.txt";
    string ffm_file = kTestfilename + "_ffm.txt";
    RemoveFile(lr_file.c_str());
    RemoveFile(ffm_file.c_str());
  }
};

void CheckLR(const DMatrix* matrix) {
  EXPECT_EQ(matrix->row_len, kNumSamples);
  // check the first element
  EXPECT_EQ(matrix->Y[0], (real_t)0);
  EXPECT_EQ(matrix->row[0]->X[0], (real_t)1.0);
  EXPECT_EQ(matrix->row[0]->X[1], (real_t)0.123);
  EXPECT_EQ(matrix->row[0]->X[kFeatureNum], (real_t)0.123);
  EXPECT_EQ(matrix->row[0]->idx[0], (index_t)0);
  EXPECT_EQ(matrix->row[0]->idx[1], (index_t)1);
  EXPECT_EQ(matrix->row[0]->idx[kFeatureNum], (index_t)(kFeatureNum));
  // check the last element
  EXPECT_EQ(matrix->Y[kNumSamples-1], (real_t)0);
  EXPECT_EQ(matrix->row[kNumSamples-1]->X[0], (real_t)1.0);
  EXPECT_EQ(matrix->row[kNumSamples-1]->X[1], (real_t)0.123);
  EXPECT_EQ(matrix->row[kNumSamples-1]->X[kFeatureNum], (real_t)0.123);
  EXPECT_EQ(matrix->row[kNumSamples-1]->idx[0], (index_t)0);
  EXPECT_EQ(matrix->row[kNumSamples-1]->idx[1], (index_t)1);
  EXPECT_EQ(matrix->row[kNumSamples-1]->idx[kFeatureNum], (index_t)(kFeatureNum));
}

void CheckFFM(const DMatrix* matrix) {
  EXPECT_EQ(matrix->row_len, kNumSamples);
  // check the first element
  EXPECT_EQ(matrix->Y[0], (real_t)1);
  EXPECT_EQ(matrix->row[0]->X[0], (real_t)1.0);
  EXPECT_EQ(matrix->row[0]->X[1], (real_t)0.123);
  EXPECT_EQ(matrix->row[0]->X[kFeatureNum], (real_t)0.123);
  EXPECT_EQ(matrix->row[0]->idx[0], (index_t)0);
  EXPECT_EQ(matrix->row[0]->idx[1], (index_t)1);
  EXPECT_EQ(matrix->row[0]->idx[kFeatureNum], (index_t)(kFeatureNum));
  EXPECT_EQ(matrix->row[0]->field[0], (index_t)0);
  EXPECT_EQ(matrix->row[0]->field[1], (index_t)1);
  EXPECT_EQ(matrix->row[0]->field[kFeatureNum], (index_t)(kFeatureNum));
  // check the last element
  EXPECT_EQ(matrix->Y[kNumSamples-1], (real_t)1);
  EXPECT_EQ(matrix->row[kNumSamples-1]->X[0], (real_t)1.0);
  EXPECT_EQ(matrix->row[kNumSamples-1]->X[1], (real_t)0.123);
  EXPECT_EQ(matrix->row[kNumSamples-1]->X[kFeatureNum], (real_t)0.123);
  EXPECT_EQ(matrix->row[kNumSamples-1]->idx[0], (index_t)0);
  EXPECT_EQ(matrix->row[kNumSamples-1]->idx[1], (index_t)1);
  EXPECT_EQ(matrix->row[kNumSamples-1]->idx[kFeatureNum], (index_t)(kFeatureNum));
  EXPECT_EQ(matrix->row[kNumSamples-1]->field[0], (index_t)0);
  EXPECT_EQ(matrix->row[kNumSamples-1]->field[1], (index_t)1);
  EXPECT_EQ(matrix->row[kNumSamples-1]->field[kFeatureNum], (index_t)(kFeatureNum));
}

TEST_F(ReaderTest, SampleFromMemory) {
  string lr_file = kTestfilename + "_LR.txt";
  string ffm_file = kTestfilename + "_ffm.txt";
  // lr
  InmemReader reader_lr;
  reader_lr.Initialize(lr_file, kNumSamples, parser_lr, LR);
  DMatrix* matrix = NULL;
  int i = 0;
  for (; i < iteration_num; ++i) {
    int record_num = reader_lr.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader_lr.GoToHead();
      continue;
    }
    EXPECT_EQ(record_num, kNumSamples);
    CheckLR(matrix);
  }
  EXPECT_EQ(i, iteration_num);
  // ffm
  matrix = nullptr;
  InmemReader reader_ffm;
  reader_ffm.Initialize(ffm_file, kNumSamples, parser_ffm, FFM);
  for (i = 0; i < iteration_num; ++i) {
    int record_num = reader_ffm.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader_ffm.GoToHead();
      continue;
    }
    EXPECT_EQ(record_num, kNumSamples);
    CheckFFM(matrix);
  }
  EXPECT_EQ(i, iteration_num);
}

TEST_F(ReaderTest, SampleFromDisk) {
  string lr_file = kTestfilename + "_LR.txt";
  string ffm_file = kTestfilename + "_ffm.txt";
  // lr
  OndiskReader reader_lr;
  reader_lr.Initialize(lr_file, kNumSamples, parser_lr, LR);
  DMatrix* matrix = nullptr;
  int i = 0;
  for (; i < iteration_num; ++i) {
    int record_num = reader_lr.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader_lr.GoToHead();
      continue;
    }
    EXPECT_EQ(record_num, kNumSamples);
    CheckLR(matrix);
  }
  EXPECT_EQ(i, iteration_num);
  // ffm
  OndiskReader reader_ffm;
  reader_ffm.Initialize(ffm_file, kNumSamples, parser_ffm, FFM);
  for (i = 0; i < iteration_num; ++i) {
    int record_num = reader_ffm.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader_ffm.GoToHead();
      continue;
    }
    EXPECT_EQ(record_num, kNumSamples);
    CheckFFM(matrix);
  }
  EXPECT_EQ(i, iteration_num);
}

Reader* CreateReader(const char* format_name) {
  return CREATE_READER(format_name);
}

TEST(READER_TEST, CreateReader) {
  EXPECT_TRUE(CreateReader("memory") != NULL);
  EXPECT_TRUE(CreateReader("disk") != NULL);
  EXPECT_TRUE(CreateReader("") == NULL);
  EXPECT_TRUE(CreateReader("unknow_name") == NULL);
}

} // namespace f2m
