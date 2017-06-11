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

This file tests data_structure.h
*/

#include "gtest/gtest.h"

#include "src/data/data_structure.h"

namespace f2m {

TEST(SPARSE_ROW_TEST, Init) {
  SparseRow row(10, true);
  EXPECT_EQ(row.column_len, 10);
  EXPECT_EQ(row.X.size(), 10);
  EXPECT_EQ(row.idx.size(), 10);
  EXPECT_EQ(row.field.size(), 10);
  EXPECT_EQ(row.if_has_field, true);
}

TEST(SPARSE_ROW_TEST, Resize) {
  SparseRow row(10, true);
  row.Resize(20);
  EXPECT_EQ(row.column_len, 20);
  EXPECT_EQ(row.X.size(), 20);
  EXPECT_EQ(row.idx.size(), 20);
  EXPECT_EQ(row.field.size(), 20);
  EXPECT_EQ(row.if_has_field, true);
  row.Resize(10);
  EXPECT_EQ(row.column_len, 10);
  EXPECT_EQ(row.X.size(), 20);
  EXPECT_EQ(row.idx.size(), 20);
  EXPECT_EQ(row.field.size(), 20);
  EXPECT_EQ(row.if_has_field, true);
}

TEST(SPARSE_ROW_TEST, CopyFrom) {
  SparseRow row(10, true);
  for (size_t i = 0; i < 10; ++i) {
    row.X[i] = 3;
    row.idx[i] = i;
    row.field[i] = i;
  }
  SparseRow row_2(5, true);
  EXPECT_EQ(row_2.column_len, 5);
  EXPECT_EQ(row_2.X.size(), 5);
  EXPECT_EQ(row_2.idx.size(), 5);
  EXPECT_EQ(row_2.field.size(), 5);
  EXPECT_EQ(row.if_has_field, true);
  row_2.CopyFrom(&row);
  EXPECT_EQ(row_2.column_len, 10);
  EXPECT_EQ(row_2.X.size(), 10);
  EXPECT_EQ(row_2.idx.size(), 10);
  EXPECT_EQ(row_2.field.size(), 10);
  EXPECT_EQ(row.if_has_field, true);
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(row_2.X[i], 3);
    EXPECT_EQ(row_2.idx[i], i);
    EXPECT_EQ(row_2.field[i], i);
  }
}

TEST(DMATRIX_TEST, Init) {
  DMatrix matrix(10);
  EXPECT_EQ(matrix.row.size(), 10);
  EXPECT_EQ(matrix.Y.size(), 10);
  EXPECT_EQ(matrix.row_len, 10);
  EXPECT_EQ(matrix.can_release, false);
}

TEST(DMATRIX_TEST, Resize) {
  DMatrix matrix(10);
  matrix.Resize(20);
  EXPECT_EQ(matrix.row.size(), 20);
  EXPECT_EQ(matrix.Y.size(), 20);
  EXPECT_EQ(matrix.row_len, 20);
  EXPECT_EQ(matrix.can_release, false);
  matrix.Resize(15);
  EXPECT_EQ(matrix.row.size(), 15);
  EXPECT_EQ(matrix.Y.size(), 15);
  EXPECT_EQ(matrix.row_len, 15);
  EXPECT_EQ(matrix.can_release, false);
}

TEST(DMATRIX_TEST, InitSparseRow) {
  DMatrix matrix(10);
  matrix.Resize(20);
  matrix.InitSparseRow(true);
  EXPECT_EQ(matrix.row.size(), 20);
  EXPECT_EQ(matrix.Y.size(), 20);
  EXPECT_EQ(matrix.row_len, 20);
  EXPECT_EQ(matrix.can_release, true);
}

TEST(DMATRIX_TEST, CopyFrom) {
  DMatrix matrix(20);
  matrix.InitSparseRow(true);
  DMatrix matrix_2(15);
  matrix_2.InitSparseRow(true);
  matrix.CopyFrom(matrix_2);
  EXPECT_EQ(matrix.row.size(), 15);
  EXPECT_EQ(matrix.Y.size(), 15);
  EXPECT_EQ(matrix.row_len, 15);
  EXPECT_EQ(matrix.can_release, true);
}

} // namespace f2m
