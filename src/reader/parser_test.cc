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

using std::vector;
using std::string;

namespace xLearn {

const string kStr = "0 1:0.123 2:0.123 3:0.123 4:0.123 5:0.123";
const string kStrFFM = "1 1:1:0.123 2:2:0.123 3:3:0.123 4:4:0.123 5:5:0.123";
const string kStrCSV = "0 1 2 3 4 5";
const index_t kNum_lines = 100000;
const index_t kLen = 5;

TEST(PARSER_TEST, Parse_libsvm) {
  StringList list(kNum_lines);
  for (index_t i = 0; i < kNum_lines; ++i) {
    list[i] = kStr;
  }
  DMatrix matrix(kNum_lines);
  matrix.InitSparseRow();
  LibsvmParser parser;
  parser.Parse(list, matrix);
  EXPECT_EQ(matrix.row_len, kNum_lines);
  EXPECT_EQ(matrix.row.size(), kNum_lines);
  EXPECT_EQ(matrix.Y.size(), kNum_lines);
  for (index_t i = 0; i < kNum_lines; ++i) {
    EXPECT_EQ(matrix.row[i]->X.size(), kLen + 1);
    EXPECT_EQ(matrix.row[i]->idx.size(), kLen + 1);
    EXPECT_EQ(matrix.Y[i], (real_t(0)));
    for (index_t j = 1; j <= kLen; ++j) {
      EXPECT_EQ(matrix.row[i]->X[j], (real_t)(0.123));
      EXPECT_EQ(matrix.row[i]->idx[j], j);
    }
    EXPECT_EQ(matrix.row[i]->X[0], 1.0);
    EXPECT_EQ(matrix.row[i]->idx[0], 0);
  }
  matrix.Release();
}

TEST(PARSER_TEST, Parse_libffm) {
  StringList list(kNum_lines);
  for (index_t i = 0; i < kNum_lines; ++i) {
    list[i] = kStrFFM;
  }
  DMatrix matrix(kNum_lines);
  matrix.InitSparseRow(true); // true for ffm
  FFMParser parser;
  parser.Parse(list, matrix);
  EXPECT_EQ(matrix.row_len, kNum_lines);
  EXPECT_EQ(matrix.row.size(), kNum_lines);
  EXPECT_EQ(matrix.Y.size(), kNum_lines);
  for (index_t i = 0; i < kNum_lines; ++i) {
    EXPECT_EQ(matrix.row[i]->X.size(), kLen + 1);
    EXPECT_EQ(matrix.row[i]->idx.size(), kLen + 1);
    EXPECT_EQ(matrix.row[i]->field.size(), kLen + 1);
    EXPECT_EQ(matrix.Y[i], (real_t)(1));
    for (index_t j = 1; j <= kLen; ++j) {
      EXPECT_EQ(matrix.row[i]->X[j], (real_t(0.123)));
      EXPECT_EQ(matrix.row[i]->idx[j], j);
      EXPECT_EQ(matrix.row[i]->field[j], j);
    }
    EXPECT_EQ(matrix.row[i]->X[0], 1.0);
    EXPECT_EQ(matrix.row[i]->idx[0], 0);
    EXPECT_EQ(matrix.row[i]->field[0], 0);
  }
  matrix.Release();
}

TEST(PARSER_TEST, Parse_csv) {
  StringList list(kNum_lines);
  for (index_t i = 0; i < kNum_lines; ++i) {
    list[i] = kStrCSV;
  }
  DMatrix matrix(kNum_lines);
  matrix.InitSparseRow(false); // false : no field
  CSVParser parser;
  parser.Parse(list, matrix);
  EXPECT_EQ(matrix.row_len, kNum_lines);
  EXPECT_EQ(matrix.row.size(), kNum_lines);
  EXPECT_EQ(matrix.Y.size(), kNum_lines);
  for (index_t i = 0; i < kNum_lines; ++i) {
    EXPECT_EQ(matrix.row[i]->X.size(), kLen + 1);
    EXPECT_EQ(matrix.row[i]->idx.size(), kLen + 1);
    EXPECT_EQ(matrix.Y[i], real_t(0));
    for (index_t j = 1; j <= kLen; ++j) {
      EXPECT_EQ(matrix.row[i]->X[j], (real_t(j)));
      EXPECT_EQ(matrix.row[i]->idx[j], j);
    }
    EXPECT_EQ(matrix.row[i]->X[0], 1.0);
    EXPECT_EQ(matrix.row[i]->idx[0], 0);
  }
  matrix.Release();
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
