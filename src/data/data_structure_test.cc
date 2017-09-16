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

namespace xLearn {

TEST(DMATRIX_TEST, Resize_and_Release) {
  DMatrix matrix;
  matrix.ResetMatrix(10);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(matrix.row[i], nullptr);
    EXPECT_EQ(matrix.Y[i], 0);
  }
  matrix.Release();
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(matrix.row.empty(), true);
    EXPECT_EQ(matrix.Y.empty(), true);
  }
}

TEST(DMATRIX_TEST, Serialize_and_Deserialize) {
  DMatrix matrix;
  matrix.ResetMatrix(10);
  for (int i = 0; i < 10; ++i) {
    matrix.row[i] = new SparseRow;
    matrix.AddNode(i, i, 2.5, i);
    matrix.Y[i] = i;
  }
  matrix.Serialize("/tmp/test.bin");
  matrix.Release();
  matrix.Deserialize("/tmp/test.bin");
  EXPECT_EQ(matrix.row_length, 10);
  EXPECT_EQ(matrix.hash_value_1, 0);
  EXPECT_EQ(matrix.hash_value_2, 0);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(matrix.Y[i], i);
    SparseRow *row = matrix.row[i];
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      EXPECT_EQ(iter->field_id, i);
      EXPECT_EQ(iter->feat_id, i);
      EXPECT_FLOAT_EQ(iter->feat_val, 2.5);
    }
  }
  RemoveFile("/tmp/test.bin");
}

}  // namespace xLearn
