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
This file tests data_structure.h file.
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

TEST(DMATRIX_TEST, Compress) {
  // Init matrix
  DMatrix matrix;
  matrix.ResetMatrix(4);
  // row_0
  matrix.AddNode(0, 1, 0.1);
  matrix.AddNode(0, 5, 0.1);
  matrix.AddNode(0, 8, 0.1);
  matrix.AddNode(0, 10, 0.1);
  // row_1
  matrix.AddNode(1, 3, 0.1);
  matrix.AddNode(1, 12, 0.1);
  matrix.AddNode(1, 20, 0.1);
  // row_2
  matrix.AddNode(2, 5, 0.1);
  matrix.AddNode(2, 8, 0.1);
  matrix.AddNode(2, 11, 0.1);
  // row_3
  matrix.AddNode(3, 2, 0.1);
  matrix.AddNode(3, 4, 0.1);
  matrix.AddNode(3, 7, 0.1);
  // Compress
  std::vector<index_t> feature_list;
  matrix.Compress(feature_list);
  // row_0
  SparseRow* row = matrix.row[0];
  EXPECT_EQ((*row)[0].feat_id, 1);
  EXPECT_EQ((*row)[1].feat_id, 5);
  EXPECT_EQ((*row)[2].feat_id, 7);
  EXPECT_EQ((*row)[3].feat_id, 8);
  // row_1
  row = matrix.row[1];
  EXPECT_EQ((*row)[0].feat_id, 3);
  EXPECT_EQ((*row)[1].feat_id, 10);
  EXPECT_EQ((*row)[2].feat_id, 11);
  // row_2
  row = matrix.row[2];
  EXPECT_EQ((*row)[0].feat_id, 5);
  EXPECT_EQ((*row)[1].feat_id, 7);
  EXPECT_EQ((*row)[2].feat_id, 9);
  // row_3
  row = matrix.row[3];
  EXPECT_EQ((*row)[0].feat_id, 2);
  EXPECT_EQ((*row)[1].feat_id, 4);
  EXPECT_EQ((*row)[2].feat_id, 6);
  // feature list
  EXPECT_EQ(feature_list[0], 1);
  EXPECT_EQ(feature_list[1], 2);
  EXPECT_EQ(feature_list[2], 3);
  EXPECT_EQ(feature_list[3], 4);
  EXPECT_EQ(feature_list[4], 5);
  EXPECT_EQ(feature_list[5], 7);
  EXPECT_EQ(feature_list[6], 8);
  EXPECT_EQ(feature_list[7], 10);
  EXPECT_EQ(feature_list[8], 11);
  EXPECT_EQ(feature_list[9], 12);
  EXPECT_EQ(feature_list[10], 20);
}

TEST(DMATRIX_TEST, GetMiniBatch) {
  // Init matrix
  DMatrix matrix;
  matrix.ResetMatrix(10);
  for (size_t i = 0; i < 10; ++i) {
    matrix.AddNode(i, i, 2.5, i);
    matrix.Y[i] = i;
    matrix.norm[i] = 0.25;
  }
  DMatrix mini_batch;
  mini_batch.ResetMatrix(4);
  index_t res = 0;
  // Get mini-batch (4 samples)
  res = matrix.GetMiniBatch(4, mini_batch);
  EXPECT_EQ(res, 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(mini_batch.Y[i], i);
    EXPECT_EQ(mini_batch.norm[i], 0.25);
    SparseRow *row =mini_batch.row[i];
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      EXPECT_EQ(iter->field_id, i);
      EXPECT_EQ(iter->feat_id, i);
      EXPECT_FLOAT_EQ(iter->feat_val, 2.5);
    }
  }
  // Get mini-batch (4 samples)
  res = matrix.GetMiniBatch(4, mini_batch);
  EXPECT_EQ(res, 4);
  for (int i = 4; i < 8; ++i) {
    EXPECT_EQ(mini_batch.Y[i-4], i);
    EXPECT_EQ(mini_batch.norm[i-4], 0.25);
    SparseRow *row =mini_batch.row[i-4];
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      EXPECT_EQ(iter->field_id, i);
      EXPECT_EQ(iter->feat_id, i);
      EXPECT_FLOAT_EQ(iter->feat_val, 2.5);
    }
  }
  // Get mini-batch (2 samples)
  res = matrix.GetMiniBatch(4, mini_batch);
  EXPECT_EQ(res, 2);
  for (int i = 8; i < 10; ++i) {
    EXPECT_EQ(mini_batch.Y[i-8], i);
    EXPECT_EQ(mini_batch.norm[i-8], 0.25);
    SparseRow *row =mini_batch.row[i-8];
    for (SparseRow::iterator iter = row->begin();
         iter != row->end(); ++iter) {
      EXPECT_EQ(iter->field_id, i);
      EXPECT_EQ(iter->feat_id, i);
      EXPECT_FLOAT_EQ(iter->feat_val, 2.5);
    }
  }
  // Get mini-batch (0 samples)
  res = matrix.GetMiniBatch(4, mini_batch);
  EXPECT_EQ(res, 0);
  // Get mini-batch (0 samples)
  res = matrix.GetMiniBatch(4, mini_batch);
  EXPECT_EQ(res, 0);
}

}  // namespace xLearn
