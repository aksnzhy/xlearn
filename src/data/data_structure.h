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

This file defines the basic data structures used by xLearn.
*/

#ifndef XLEARN_DATA_DATA_STRUCTURE_H_
#define XLEARN_DATA_DATA_STRUCTURE_H_

#include <vector>

#include "src/base/common.h"
#include "src/base/file_util.h"

namespace xLearn {

//------------------------------------------------------------------------------
// We use 32 bits float to store the real number, such as the model
// parameter and the gradient entry
//------------------------------------------------------------------------------
typedef float real_t;

//------------------------------------------------------------------------------
// We use 32 bits unsigned int to store feature index
//------------------------------------------------------------------------------
typedef uint32 index_t;

//------------------------------------------------------------------------------
// We use SSE to speedup our training, so some hyper
// parameters should be aligned
//------------------------------------------------------------------------------
const int kAlign = 4;
const int kAlignByte = 16;

//------------------------------------------------------------------------------
// Node is used to store information for each feature
// For tasks like lr and fm, we just need to store the feature id
// and feature value, while we also need to store the field id for
// the ffm task
//------------------------------------------------------------------------------
struct Node {
  index_t field_id;  /* Start from 0 */
  index_t feat_id;   /* Start from 0, which is the bias term */
  real_t feat_val;   /* Can be numeric or catagorical feature */
};

//------------------------------------------------------------------------------
// SparseRow is used to store one line of the training data, which
// is a vector of the Node structure
//------------------------------------------------------------------------------
typedef std::vector<Node> SparseRow;

//------------------------------------------------------------------------------
// DMatrix (data matrix) is used to store a batch of training dataset.
// It can be the whole data set used in in-memory training, and or working
// set in on-disk training, because for many large-scale ML problems, we
// cannot load all the training data into memory at once. So we can load a
// small batch of dataset into the DMatrix at each iteration.
// We can use the DMatrix like this:
//
//    DMatrix matrix;
//    matrix.ResetMatrix(10);   /* Init 10 rows */
//    for (int i = 0; i < 10; ++i) {
//      matrix.Y[i] = ...  /* set y */
//      matrix.row[i] = new SparseRow;
//      matrix.AddNode(i, feat_id, feat_val, field_id);
//    }
//    matrix.Serialize("/tmp/test.bin");    /* Serialize matrix to file */
//    matrix.Release();
//    matrix.Deserialize("/tmp/test.bin");  /* Deserialize matrix from file */
//
//    /* We can access the matrix like this */
//    for (int i = 0; i < matrix.row_length; ++i) {
//      ... matrix.Y[i] ..   /* access y */
//      SparseRow *row = matrix.row[i];
//      for (SparseRow::iterator iter = row->begin();
//           iter != row->end(); ++iter) {
//        ... iter->field_id ...   /* access field_id */
//        ... iter->feat_id ...    /* access feat_id */
//        ... iter->feat_val ...   /* access feat_val */
//      }
//    }
//------------------------------------------------------------------------------
struct DMatrix {
  // Constructor and Destructor
  DMatrix() { }
  ~DMatrix() { Release(); }

  // Reset memory for the DMatrix
  // This function will first release the original
  // memory of the DMatrix, and then re-allocate memory
  // for that. In this function, Y will be initialized
  // to 0 and row will be initialized to a NULL pointer
  void ResetMatrix(index_t length) {
    CHECK_GE(length, 0);
    this->Release();
    row_length = length;
    row.resize(length, nullptr);
    Y.resize(length, 0);
    // we set norm to 1.0 by default, which means
    // that we don't use normalization
    norm.resize(length, 1.0);
  }

  // Release memory for DMatrix
  // Note that a typical alternative that forces a
  // reallocation is to use swap(), instead of using clear()
  void Release() {
    row_length = 0;
    // Delete Y
    std::vector<real_t>().swap(Y);
    for (int i = 0; i < row_length; ++i) {
      // Delete Node
      std::vector<Node>().swap(*row[i]);
    }
    // Delete row
    std::vector<SparseRow*>().swap(row);
    // Delete norm
    std::vector<real_t>().swap(norm);
  }

  // Add node to matrix
  // We don't use the 'field' by default and it
  // will only be used in the ffm tasks
  void AddNode(index_t row_id,  index_t feat_id,
               real_t feat_val, index_t field_id = 0) {
    CHECK_GT(row_length, row_id);
    CHECK_NOTNULL(row[row_id]);
    Node node;
    node.field_id = field_id;
    node.feat_id = feat_id;
    node.feat_val = feat_val;
    row[row_id]->push_back(node);
  }

  // The hash value is used for the Reader class, which will
  // identify the difference between two data matrix
  // The hash value is generated by HashFile() method in file_util.h
  void SetHash(uint64 hash_1, uint64 hash_2) {
    hash_value_1 = hash_1;
    hash_value_2 = hash_2;
  }

  // Serialize current DMatrix to disk file
  void Serialize(const std::string& filename) {
    CHECK(!filename.empty());
    CHECK_EQ(row_length, row.size());
    CHECK_EQ(row_length, Y.size());
    FILE* file = OpenFileOrDie(filename.c_str(), "w");
    // Write hash_value
    WriteDataToDisk(file,
      (char*)&hash_value_1, sizeof(hash_value_1));
    WriteDataToDisk(file,
      (char*)&hash_value_2, sizeof(hash_value_2));
    // Write row_length
    WriteDataToDisk(file,
      (char*)&row_length, sizeof(row_length));
    // Write row
    for (int i = 0; i < row_length; ++i) {
      WriteVectorToFile(file, *(row[i]));
    }
    // Write Y
    WriteVectorToFile(file, Y);
    // Write norm
    WriteVectorToFile(file, norm);
    Close(file);
  }

  // Deserialize the DMatrix from disk file
  void Deserialize(const std::string& filename) {
    CHECK(!filename.empty());
    this->Release();
    FILE* file = OpenFileOrDie(filename.c_str(), "r");
    // Read hash_value
    ReadDataFromDisk(file,
      (char*)&hash_value_1, sizeof(hash_value_1));
    ReadDataFromDisk(file,
      (char*)&hash_value_2, sizeof(hash_value_2));
    // Read row_length
    ReadDataFromDisk(file,
      (char*)&row_length, sizeof(row_length));
    // Read row
    row.resize(row_length);
    for (int i = 0; i < row_length; ++i) {
      row[i] = new SparseRow;
      ReadVectorFromFile(file, *(row[i]));
    }
    // Read Y
    ReadVectorFromFile(file, Y);
    // Read norm
    ReadVectorFromFile(file, norm);
    Close(file);
  }

  /* The DMatrix has a hash value that is
  geneerated from the txt file.
  These two values are used to check whether
  we can use binary file to speedup data reading */
  uint64 hash_value_1;
  uint64 hash_value_2;
  /* Row length of current matrix */
  index_t row_length;
  /* Using pointer to implement zero-copy */
  std::vector<SparseRow*> row;
  /* 0 or -1 for negative and +1 for positive
  example, and others for regression */
  std::vector<real_t> Y;
  /* Used for instance-wise normalization */
  std::vector<real_t> norm;
};

}  // namespace xLearn

#endif  // XLEARN_DATA_DATA_STRUCTURE_H_
