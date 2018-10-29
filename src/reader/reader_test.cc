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
const string kStr_comma = "0,1:0.123,1:0.123,1:0.123\n";

const string kStrFFM = "1 1:1:0.123 1:1:0.123 1:1:0.123\n";
const string kStrFFM_comma = "1,1:1:0.123,1:1:0.123,1:1:0.123\n";

const string kStrCSV = "0 0.123 0.123 0.123\n";
const string kStrCSV_comma = "0,0.123,0.123,0.123\n";

const string kStrNoy = "1:0.123 1:0.123 1:0.123\n";
const string kStrNoy_comma = "1:0.123,1:0.123,1:0.123\n";

const string kStrFFMNoy = "1:1:0.123 1:1:0.123 1:1:0.123\n";
const string kStrFFMNoy_comma = "1:1:0.123,1:1:0.123,1:1:0.123\n";

const index_t kNumLines = 100000;
const int iteration_num = 10;

void write_data(const std::string& filename,
                const std::string& data) {
  FILE* file = OpenFileOrDie(filename.c_str(), "w");
  for (index_t i = 0; i < kNumLines; ++i) {
    uint32 write_len = fwrite(data.c_str(), 1, data.size(), file);
    EXPECT_EQ(write_len, data.size());
  }
  Close(file);
}

void WriteFile() {
  // has label
  string lr_file = kTestfilename + "_LR.txt";
  string ffm_file = kTestfilename + "_ffm.txt";
  string csv_file = kTestfilename + "_csv.txt";
  string lr_file_comma = kTestfilename + "_LR_comma.txt";
  string ffm_file_comma = kTestfilename + "_ffm_comma.txt";
  string csv_file_comma = kTestfilename + "_csv_comma.txt";
  // has no label
  string lr_no_file = kTestfilename + "_LR_no.txt";
  string ffm_no_file = kTestfilename + "_ffm_no.txt";
  string lr_no_file_comma = kTestfilename + "_LR_no_comma.txt";
  string ffm_no_file_comma = kTestfilename + "_ffm_no_comma.txt";
  // Create file
  write_data(lr_file, kStr);
  write_data(ffm_file, kStrFFM);
  write_data(csv_file, kStrCSV);
  write_data(lr_no_file, kStrNoy);
  write_data(ffm_no_file, kStrFFMNoy);
  write_data(lr_file_comma, kStr);
  write_data(ffm_file_comma, kStrFFM);
  write_data(csv_file_comma, kStrCSV);
  write_data(lr_no_file_comma, kStrNoy);
  write_data(ffm_no_file_comma, kStrFFMNoy);
}

void delete_file() {
  // has label
  string lr_file = kTestfilename + "_LR.txt";
  string ffm_file = kTestfilename + "_ffm.txt";
  string csv_file = kTestfilename + "_csv.txt";
  string lr_file_comma = kTestfilename + "_LR_comma.txt";
  string ffm_file_comma = kTestfilename + "_ffm_comma.txt";
  string csv_file_comma = kTestfilename + "_csv_comma.txt";
  // has no label
  string lr_no_file = kTestfilename + "_LR_no.txt";
  string ffm_no_file = kTestfilename + "_ffm_no.txt";
  string lr_no_file_comma = kTestfilename + "_LR_no_comma.txt";
  string ffm_no_file_comma = kTestfilename + "_ffm_no_comma.txt";
  // remove
  RemoveFile(lr_file.c_str());
  RemoveFile(ffm_file.c_str());
  RemoveFile(csv_file.c_str());
  RemoveFile(lr_no_file.c_str());
  RemoveFile(ffm_no_file.c_str());
  RemoveFile(lr_file_comma.c_str());
  RemoveFile(ffm_file_comma.c_str());
  RemoveFile(csv_file_comma.c_str());
  RemoveFile(lr_no_file_comma.c_str());
  RemoveFile(ffm_no_file_comma.c_str());
  // bin file
  lr_file += ".bin";
  ffm_file += ".bin";
  csv_file += ".bin";
  lr_no_file += ".bin";
  ffm_no_file += ".bin";
  lr_file_comma += ".bin";
  ffm_file_comma += ".bin";
  csv_file_comma += ".bin";
  lr_no_file_comma += ".bin";
  ffm_no_file_comma += ".bin";
  // remove
  RemoveFile(lr_file.c_str());
  RemoveFile(ffm_file.c_str());
  RemoveFile(csv_file.c_str());
  RemoveFile(lr_no_file.c_str());
  RemoveFile(ffm_no_file.c_str());
  RemoveFile(lr_file_comma.c_str());
  RemoveFile(ffm_file_comma.c_str());
  RemoveFile(csv_file_comma.c_str());
  RemoveFile(lr_no_file_comma.c_str());
  RemoveFile(ffm_no_file_comma.c_str());
}

void CheckLR(const DMatrix* matrix, bool has_label, bool disk) {
  if (disk) {
    EXPECT_EQ(matrix->row_length, kNumLines);
  } else {
    EXPECT_EQ(matrix->row_length, kNumLines);
  }
  // check the first element
  if (has_label) {
    EXPECT_EQ(matrix->Y[0], 0);
  } else {
    EXPECT_EQ(matrix->Y[0], -2);
  }
  EXPECT_FLOAT_EQ(matrix->norm[0], 22.03274);
  for (int i = 0; i < matrix->row_length; ++i) {
    SparseRow *row = matrix->row[i];
    int n = 0;
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      EXPECT_EQ(iter->field_id, 0);
      EXPECT_EQ(iter->feat_id, 1);
      EXPECT_FLOAT_EQ(iter->feat_val, 0.123);
      n++;
    }
    EXPECT_EQ(n, 3);
  }
}

void CheckFFM(const DMatrix* matrix, bool has_label, bool disk) {
  if (disk) {
    EXPECT_EQ(matrix->row_length, kNumLines);
  } else {
    EXPECT_EQ(matrix->row_length, kNumLines);
  }
  // check the first element
  if (has_label) {
    EXPECT_EQ(matrix->Y[0], 1);
  } else {
    EXPECT_EQ(matrix->Y[0], -2);
  }
  EXPECT_FLOAT_EQ(matrix->norm[0], 22.03274);
  for (int i = 0; i < matrix->row_length; ++i) {
    SparseRow *row = matrix->row[i];
    int n = 0;
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      EXPECT_EQ(iter->field_id, 1);
      EXPECT_EQ(iter->feat_id, 1);
      EXPECT_FLOAT_EQ(iter->feat_val, 0.123);
      n++;
    }
    EXPECT_EQ(n, 3);
  }
}

void CheckCSV(const DMatrix* matrix, bool disk) {
  if (disk) {
    EXPECT_EQ(matrix->row_length, kNumLines);
  } else {
    EXPECT_EQ(matrix->row_length, kNumLines);
  }
  EXPECT_EQ(matrix->Y[0], 0);
  EXPECT_FLOAT_EQ(matrix->norm[0], 22.03274);
  for (int i = 0; i < matrix->row_length; ++i) {
    SparseRow *row = matrix->row[i];
    int n = 0;
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      EXPECT_EQ(iter->feat_id, n);
      EXPECT_FLOAT_EQ(iter->feat_val, 0.123);
      n++;
    }
    EXPECT_EQ(n, 3);
  }
}

void read_from_memory(const std::string& filename, int task_id) {
  InmemReader in_mem_reader;
  Reader* reader = nullptr;
  in_mem_reader.Initialize(filename);
  reader = &in_mem_reader;
  DMatrix* matrix = nullptr;
  for (int i = 0; i < iteration_num; ++i) {
    int record_num = reader->Samples(matrix);
    if (record_num == 0) {
      --i;
      reader->Reset();
      continue;
    }
    switch (task_id) {
      case 0:
        CheckLR(matrix, true, false);
        break;
      case 1:
        CheckFFM(matrix, true, false);
        break;
      case 2:
        CheckCSV(matrix, false);
        break;
      case 3:
        CheckLR(matrix, false, false);
        break;
      case 4:
        CheckFFM(matrix, false, false);
        break;
    }
  }
}

void read_from_disk(const std::string& filename, int task_id) {
  OndiskReader reader;
  reader.SetBlockSize(100);
  reader.Initialize(filename);
  DMatrix* matrix = new DMatrix;
  for (int i = 0; i < iteration_num; ++i) {
    int record_num = reader.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader.Reset();
      continue;
    }
    switch(task_id) {
      case 0:
        CheckLR(matrix, true, true);
        break;
      case 1:
        CheckFFM(matrix, true, true);
        break;
      case 2:
        CheckCSV(matrix, true);
        break;
      case 3:
        CheckLR(matrix, false, true);
        break;
      case 4:
        CheckFFM(matrix, false, true);
        break;
    }
  } 
}

TEST(ReaderTest, SampleFromMemory) {
  WriteFile();
  // has label
  string lr_file = kTestfilename + "_LR.txt";
  string ffm_file = kTestfilename + "_ffm.txt";
  string csv_file = kTestfilename + "_csv.txt";
  string lr_file_comma = kTestfilename + "_LR_comma.txt";
  string ffm_file_comma = kTestfilename + "_ffm_comma.txt";
  string csv_file_comma = kTestfilename + "_csv_comma.txt";
  // has no label
  string lr_no_file = kTestfilename + "_LR_no_comma.txt";
  string ffm_no_file = kTestfilename + "_ffm_no_comma.txt";
  string lr_no_file_comma = kTestfilename + "_LR_no_comma.txt";
  string ffm_no_file_comma = kTestfilename + "_ffm_no_comma.txt";
  // check
  read_from_memory(lr_file, 0);
  read_from_memory(ffm_file, 1);
  read_from_memory(csv_file, 2);
  read_from_memory(lr_no_file, 3);
  read_from_memory(ffm_no_file, 4);

  read_from_memory(lr_file_comma, 0);
  read_from_memory(ffm_file_comma, 1);
  read_from_memory(csv_file_comma, 2);
  read_from_memory(lr_no_file_comma, 3);
  read_from_memory(ffm_no_file_comma, 4);
}

TEST(ReaderTest, ReadFromBinary) {
  // has label
  string lr_file = kTestfilename + "_LR.txt";
  string ffm_file = kTestfilename + "_ffm.txt";
  string csv_file = kTestfilename + "_csv.txt";
  string lr_file_comma = kTestfilename + "_LR_comma.txt";
  string ffm_file_comma = kTestfilename + "_ffm_comma.txt";
  string csv_file_comma = kTestfilename + "_csv_comma.txt";
  // has no label
  string lr_no_file = kTestfilename + "_LR_no.txt";
  string ffm_no_file = kTestfilename + "_ffm_no.txt";
  string lr_no_file_comma = kTestfilename + "_LR_no_comma.txt";
  string ffm_no_file_comma = kTestfilename + "_ffm_no_comma.txt";
  // check
  read_from_memory(lr_file, 0);
  read_from_memory(ffm_file, 1);
  read_from_memory(csv_file, 2);
  read_from_memory(lr_no_file, 3);
  read_from_memory(ffm_no_file, 4);
  read_from_memory(lr_file_comma, 0);
  read_from_memory(ffm_file_comma, 1);
  read_from_memory(csv_file_comma, 2);
  read_from_memory(lr_no_file_comma, 3);
  read_from_memory(ffm_no_file_comma, 4);  
}

TEST(ReaderTest, SampleFromDisk) { 
  // has label
  string lr_file = kTestfilename + "_LR.txt";
  string ffm_file = kTestfilename + "_ffm.txt";
  string csv_file = kTestfilename + "_csv.txt";
  string lr_file_comma = kTestfilename + "_LR_comma.txt";
  string ffm_file_comma = kTestfilename + "_ffm_comma.txt";
  string csv_file_comma = kTestfilename + "_csv_comma.txt";
  // has no label
  string lr_no_file = kTestfilename + "_LR_no.txt";
  string ffm_no_file = kTestfilename + "_ffm_no.txt";
  string lr_no_file_comma = kTestfilename + "_LR_no_comma.txt";
  string ffm_no_file_comma = kTestfilename + "_ffm_no_comma.txt";
  // check
  read_from_disk(lr_file, 0);
  printf("check lr\n");
  read_from_disk(ffm_file, 1);
  printf("check ffm\n");
  read_from_disk(csv_file, 2);
  printf("check csv\n");
  read_from_disk(lr_no_file, 3);
  printf("check lr\n");
  read_from_disk(ffm_no_file, 4);
  printf("check ffm\n");
  read_from_disk(lr_file_comma, 0);
  printf("check lr\n");
  read_from_disk(ffm_file_comma, 1);
  printf("check ffm\n");
  read_from_disk(csv_file_comma, 2);
  printf("check csv\n");
  read_from_disk(lr_no_file_comma, 3);
  printf("check lr\n");
  read_from_disk(ffm_no_file_comma, 4);
  printf("check ffm\n");
  // delete file
  delete_file();
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

} // namespace xLearn
