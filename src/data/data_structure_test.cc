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
  EXPECT_EQ(matrix.hash_value_1, 0);
  EXPECT_EQ(matrix.hash_value_2, 0);
  EXPECT_EQ(matrix.row_length, 10);
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(matrix.row[i], nullptr);
    EXPECT_FLOAT_EQ(matrix.Y[i], 0);
    EXPECT_FLOAT_EQ(matrix.norm[i], 1.0);
    EXPECT_EQ(matrix.has_label, true);
  }
  matrix.Release();
  EXPECT_EQ(matrix.row_length, 0);
  EXPECT_EQ(matrix.hash_value_1, 0);
  EXPECT_EQ(matrix.hash_value_2, 0);
  EXPECT_EQ(matrix.has_label, false);
  EXPECT_EQ(matrix.row.empty(), true);
  EXPECT_EQ(matrix.Y.empty(), true);
  EXPECT_EQ(matrix.norm.empty(), true);
}

TEST(DMATRIX_TEST, Serialize_and_Deserialize) {
  DMatrix matrix;
  // Init
  matrix.ResetMatrix(10);
  for (size_t i = 0; i < 10; ++i) {
    matrix.AddNode(i, i, 2.5, i);
    matrix.Y[i] = i;
    matrix.norm[i] = 0.25;
    matrix.has_label = false;
  }
  matrix.SetHash(1234, 5678);
  // Serialize
  matrix.Serialize("/tmp/test.bin");
  matrix.Release();
  EXPECT_EQ(matrix.row_length, 0);
  EXPECT_EQ(matrix.hash_value_1, 0);
  EXPECT_EQ(matrix.hash_value_2, 0);
  EXPECT_EQ(matrix.has_label, false);
  EXPECT_EQ(matrix.row.empty(), true);
  EXPECT_EQ(matrix.Y.empty(), true);
  EXPECT_EQ(matrix.norm.empty(), true);
  // Deserialize
  matrix.Deserialize("/tmp/test.bin");
  EXPECT_EQ(matrix.row_length, 10);
  EXPECT_EQ(matrix.hash_value_1, 1234);
  EXPECT_EQ(matrix.hash_value_2, 5678);
  EXPECT_EQ(matrix.has_label, false);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(matrix.Y[i], i);
    EXPECT_EQ(matrix.norm[i], 0.25);
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

TEST(DMATRIX_TEST, Find_max_feat_and_field) {
  DMatrix matrix;
  matrix.ResetMatrix(10);
  for (size_t i = 0; i < 10; ++i) {
    matrix.AddNode(i, i, 2.5, i);
    matrix.Y[i] = i;
    matrix.norm[i] = 0.25;
    matrix.has_label = false;
  }
  matrix.SetHash(1234, 5678);
  EXPECT_EQ(matrix.MaxFeat(), 9);
  EXPECT_EQ(matrix.MaxField(), 9);
}

TEST(DMATRIX_TEST, CopyFrom) {
  // Init matrix
  DMatrix matrix;
  matrix.ResetMatrix(10);
  for (size_t i = 0; i < 10; ++i) {
    matrix.AddNode(i, i, 2.5, i);
    matrix.Y[i] = i;
    matrix.norm[i] = 0.25;
  }
  matrix.has_label = false;
  matrix.SetHash(1234, 5678);
  // Copy matrix
  DMatrix new_matrix;
  new_matrix.CopyFrom(&matrix);
  matrix.Release();
  // Check
  EXPECT_EQ(new_matrix.row_length, 10);
  EXPECT_EQ(new_matrix.hash_value_1, 1234);
  EXPECT_EQ(new_matrix.hash_value_2, 5678);
  EXPECT_EQ(new_matrix.has_label, false);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(new_matrix.Y[i], i);
    EXPECT_EQ(new_matrix.norm[i], 0.25);
    SparseRow *row =new_matrix.row[i];
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      EXPECT_EQ(iter->field_id, i);
      EXPECT_EQ(iter->feat_id, i);
      EXPECT_FLOAT_EQ(iter->feat_val, 2.5);
    }
  }
}

}  // namespace xLearn
