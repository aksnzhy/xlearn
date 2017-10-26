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

This file tests parser.h
*/

#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "src/reader/parser.h"
#include "src/data/data_structure.h"

namespace xLearn {

const std::string kStr = "0 1:0.12 2:0.12 3:0.12 4:0.12 5:0.12\n";
const std::string kStrFFM = "1 1:1:0.13 2:2:0.13 3:3:0.13 4:4:0.13 5:5:0.13\n";
const std::string kStrCSV = "0.123 0.123 0.123 0.123 0.123 1\n";
const std::string kStrNoy = "1:0.12 2:0.12 3:0.12 4:0.12 5:0.12\n";
const std::string kStrFFMNoy = "1:1:0.13 2:2:0.13 3:3:0.13 4:4:0.13 5:5:0.13\n";
const index_t kNum_lines = 100000;
const std::string filename = "./test_file.txt";

TEST(PARSER_TEST, Parse_libsvm) {
  FILE* file = OpenFileOrDie(filename.c_str(), "w");
  for (int i = 0; i < kNum_lines; ++i) {
    WriteDataToDisk(file, kStr.data(), kStr.size());
  }
  Close(file);
  char* buffer = nullptr;
  uint64 size = ReadFileToMemory(filename, &buffer);
  DMatrix matrix;
  LibsvmParser parser;
  parser.setLabel(true);
  parser.Parse(buffer, size, matrix);
  EXPECT_EQ(matrix.row_length, kNum_lines);
  for (index_t i = 0; i < matrix.row_length; ++i) {
    EXPECT_EQ(matrix.Y[i], 0);
    EXPECT_FLOAT_EQ(matrix.norm[i], 1.6666667);
    int col_len = matrix.row[i]->size();
    EXPECT_EQ(col_len, 6);
    SparseRow *row = matrix.row[i];
    int n = 0;
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      EXPECT_EQ(iter->field_id, 0);
      EXPECT_EQ(iter->feat_id, n);
      if (n == 0) {
        EXPECT_FLOAT_EQ(iter->feat_val, 1.0);
      } else {
        EXPECT_FLOAT_EQ(iter->feat_val, 0.12);
      }
      n++;
    }
    EXPECT_EQ(n, 6);
  }
  RemoveFile(filename.c_str());
}

TEST(PARSER_TEST, Parse_libsvm_no_y) {
  FILE* file = OpenFileOrDie(filename.c_str(), "w");
  for (int i = 0; i < kNum_lines; ++i) {
    WriteDataToDisk(file, kStrNoy.data(), kStrNoy.size());
  }
  Close(file);
  char* buffer = nullptr;
  uint64 size = ReadFileToMemory(filename, &buffer);
  DMatrix matrix;
  LibsvmParser parser;
  parser.setLabel(false);
  parser.Parse(buffer, size, matrix);
  EXPECT_EQ(matrix.row_length, kNum_lines);
  for (index_t i = 0; i < matrix.row_length; ++i) {
    EXPECT_EQ(matrix.Y[i], -2);
    EXPECT_FLOAT_EQ(matrix.norm[i], 1.6666667);
    int col_len = matrix.row[i]->size();
    EXPECT_EQ(col_len, 6);
    SparseRow *row = matrix.row[i];
    int n = 0;
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      EXPECT_EQ(iter->field_id, 0);
      EXPECT_EQ(iter->feat_id, n);
      if (n == 0) {
        EXPECT_FLOAT_EQ(iter->feat_val, 1.0);
      } else {
        EXPECT_FLOAT_EQ(iter->feat_val, 0.12);
      }
      n++;
    }
    EXPECT_EQ(n, 6);
  }
  RemoveFile(filename.c_str());
}

TEST(PARSER_TEST, Parse_libffm) {
  FILE* file = OpenFileOrDie(filename.c_str(), "w");
  for (int i = 0; i < kNum_lines; ++i) {
    WriteDataToDisk(file, kStrFFM.data(), kStrFFM.size());
  }
  Close(file);
  char* buffer = nullptr;
  uint64 size = ReadFileToMemory(filename, &buffer);
  DMatrix matrix;
  FFMParser parser;
  parser.setLabel(true);
  parser.Parse(buffer, size, matrix);
  EXPECT_EQ(matrix.row_length, kNum_lines);
  for (index_t i = 0; i < matrix.row_length; ++i) {
    EXPECT_EQ(matrix.Y[i], 1);
    EXPECT_FLOAT_EQ(matrix.norm[i], 1.5384616);
    int col_len = matrix.row[i]->size();
    EXPECT_EQ(col_len, 6);
    SparseRow *row = matrix.row[i];
    int n = 0;
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      EXPECT_EQ(iter->field_id, n);
      EXPECT_EQ(iter->feat_id, n);
      if (n == 0) {
        EXPECT_FLOAT_EQ(iter->feat_val, 1.0);
      } else {
        EXPECT_FLOAT_EQ(iter->feat_val, 0.13);
      }
      n++;
    }
    EXPECT_EQ(n, 6);
  }
  RemoveFile(filename.c_str());
}

TEST(PARSER_TEST, Parse_libffm_no_y) {
  FILE* file = OpenFileOrDie(filename.c_str(), "w");
  for (int i = 0; i < kNum_lines; ++i) {
    WriteDataToDisk(file, kStrFFMNoy.data(), kStrFFMNoy.size());
  }
  Close(file);
  char* buffer = nullptr;
  uint64 size = ReadFileToMemory(filename, &buffer);
  DMatrix matrix;
  FFMParser parser;
  parser.setLabel(false);
  parser.Parse(buffer, size, matrix);
  EXPECT_EQ(matrix.row_length, kNum_lines);
  for (index_t i = 0; i < matrix.row_length; ++i) {
    EXPECT_EQ(matrix.Y[i], -2);
    EXPECT_FLOAT_EQ(matrix.norm[i], 1.5384616);
    int col_len = matrix.row[i]->size();
    EXPECT_EQ(col_len, 6);
    SparseRow *row = matrix.row[i];
    int n = 0;
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      EXPECT_EQ(iter->field_id, n);
      EXPECT_EQ(iter->feat_id, n);
      if (n == 0) {
        EXPECT_FLOAT_EQ(iter->feat_val, 1.0);
      } else {
        EXPECT_FLOAT_EQ(iter->feat_val, 0.13);
      }
      n++;
    }
    EXPECT_EQ(n, 6);
  }
  RemoveFile(filename.c_str());
}

TEST(PARSER_TEST, Parse_csv) {
  FILE* file = OpenFileOrDie(filename.c_str(), "w");
  for (int i = 0; i < kNum_lines; ++i) {
    WriteDataToDisk(file, kStrCSV.data(), kStrCSV.size());
  }
  Close(file);
  char* buffer = nullptr;
  uint64 size = ReadFileToMemory(filename, &buffer);
  DMatrix matrix;
  CSVParser parser;
  parser.setLabel(true);
  parser.Parse(buffer, size, matrix);
  EXPECT_EQ(matrix.row_length, kNum_lines);
  for (index_t i = 0; i < matrix.row_length; ++i) {
    EXPECT_EQ(matrix.Y[i], 1);
    EXPECT_FLOAT_EQ(matrix.norm[i], 1.62601626);
    int col_len = matrix.row[i]->size();
    EXPECT_EQ(col_len, 6);
    SparseRow *row = matrix.row[i];
    int n = 0;
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      EXPECT_EQ(iter->feat_id, n);
      if (n == 0) {
        EXPECT_FLOAT_EQ(iter->feat_val, 1.0);
      } else {
        EXPECT_FLOAT_EQ(iter->feat_val, 0.123);
      }
      n++;
    }
    EXPECT_EQ(n, 6);
  }
  RemoveFile(filename.c_str());
}

Parser* CreateParser(const char* format_name) {
  return CREATE_PARSER(format_name);
}

TEST(PARSER_TEST, CreateParser) {
  EXPECT_TRUE(CreateParser("libsvm") != NULL);
  EXPECT_TRUE(CreateParser("libffm") != NULL);
  EXPECT_TRUE(CreateParser("csv") != NULL);
  EXPECT_TRUE(CreateParser("") == NULL);
  EXPECT_TRUE(CreateParser("unknow_name") == NULL);
}

} // namespace xLearn
