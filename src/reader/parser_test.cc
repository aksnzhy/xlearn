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
This file tests parser.h file.
*/

#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "src/reader/parser.h"
#include "src/data/data_structure.h"

namespace xLearn {

const std::string kStr = "1 0:0.12 1:0.12 2:0.12 3:0.12 4:0.12\n";
const std::string kStrFFM = "1 0:0:0.12 1:1:0.12 2:2:0.12 3:3:0.12 4:4:0.12\n";
const std::string kStrCSV = "1 0.12 0.12 0.12 0.12 0.12\n";
const std::string kStrNoy = "0:0.12 1:0.12 2:0.12 3:0.12 4:0.12\n";
const std::string kStrFFMNoy = "0:0:0.12 1:1:0.12 2:2:0.12 3:3:0.12 4:4:0.12\n";
const std::string Kfilename = "./test_file.txt";
const index_t kNum_lines = 100000;

// Write data to disk file
void write_data(const std::string& filename,
                const std::string& data) {
  FILE* file = OpenFileOrDie(filename.c_str(), "w");
  for (int i = 0; i < kNum_lines; ++i) {
    WriteDataToDisk(file, data.data(), data.size());
  }
  Close(file);
}

// Check the parser's result
void check(const DMatrix& matrix, bool has_label, bool has_field) {
  EXPECT_EQ(matrix.row_length, kNum_lines);
  for (index_t i = 0; i < matrix.row_length; ++i) {
    if (has_label) {
      EXPECT_EQ(matrix.Y[i], 1);
    } else {
      EXPECT_EQ(matrix.Y[i], -2);
    }
    EXPECT_FLOAT_EQ(matrix.norm[i], 13.888889);
    int col_len = matrix.row[i]->size();
    EXPECT_EQ(col_len, 5);
    SparseRow *row = matrix.row[i];
    int n = 0;
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      if (has_field) {
        EXPECT_EQ(iter->field_id, n);
      } else {
        EXPECT_EQ(iter->field_id, 0);
      }
      EXPECT_EQ(iter->feat_id, n);
      EXPECT_FLOAT_EQ(iter->feat_val, 0.12);
      n++;
    }
    EXPECT_EQ(n, 5);
  }
}

TEST(PARSER_TEST, Parse_libsvm) {
  write_data(Kfilename, kStr);
  char* buffer = nullptr;
  uint64 size = ReadFileToMemory(Kfilename, &buffer);
  DMatrix matrix;
  LibsvmParser parser;
  parser.setLabel(true);
  parser.Parse(buffer, size, matrix);
  check(matrix, true, false);
  RemoveFile(Kfilename.c_str());
}

TEST(PARSER_TEST, Parse_libsvm_no_y) {
  write_data(Kfilename, kStrNoy);
  char* buffer = nullptr;
  uint64 size = ReadFileToMemory(Kfilename, &buffer);
  DMatrix matrix;
  LibsvmParser parser;
  parser.setLabel(false);
  parser.Parse(buffer, size, matrix);
  check(matrix, false, false);
  RemoveFile(Kfilename.c_str());
}

TEST(PARSER_TEST, Parse_libffm) {
  write_data(Kfilename, kStrFFM);
  char* buffer = nullptr;
  uint64 size = ReadFileToMemory(Kfilename, &buffer);
  DMatrix matrix;
  FFMParser parser;
  parser.setLabel(true);
  parser.Parse(buffer, size, matrix);
  check(matrix, true, true);
  RemoveFile(Kfilename.c_str());
}

TEST(PARSER_TEST, Parse_libffm_no_y) {
  write_data(Kfilename, kStrFFMNoy);
  char* buffer = nullptr;
  uint64 size = ReadFileToMemory(Kfilename, &buffer);
  DMatrix matrix;
  FFMParser parser;
  parser.setLabel(false);
  parser.Parse(buffer, size, matrix);
  check(matrix, false, true);
  RemoveFile(Kfilename.c_str());
}

TEST(PARSER_TEST, Parse_csv) {
  write_data(Kfilename, kStrCSV);
  char* buffer = nullptr;
  uint64 size = ReadFileToMemory(Kfilename, &buffer);
  DMatrix matrix;
  CSVParser parser;
  parser.setLabel(true);
  parser.Parse(buffer, size, matrix);
  check(matrix, true, false);
  RemoveFile(Kfilename.c_str());
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
