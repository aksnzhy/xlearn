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
#include <string>

#include "src/base/common.h"
#include "src/base/stl-util.h"
#include "src/base/file_util.h"

namespace xLearn {

//------------------------------------------------------------------------------
// We use 32 bits float to store real number, such as the model
// parameter and the gradient.
//------------------------------------------------------------------------------
typedef float real_t;

//------------------------------------------------------------------------------
// We use 32 bits unsigned int to store feature index.
//------------------------------------------------------------------------------
typedef uint32 index_t;

//------------------------------------------------------------------------------
// SparseRow is used to store one line of the training data.
// Note that we do not use map<int, float> to store sparse entry because of
// its poor performance. Instead, we use double-vector, where the feature value
// and its coresponding index are stored in two vectors. For the FFM task, we
// also need another vector to store the field entry.
//------------------------------------------------------------------------------
struct SparseRow {
  // On default the 'field' vector is empty.
  explicit SparseRow(size_t length, bool has_field = false)
    : X(length, 0.0), idx(length, 0), if_has_field(has_field) {
    // for ffm task
    if (if_has_field) {
      field.resize(length, 0);
    }
    column_len = length;
  }

  // Resize current row.
  void Resize(size_t new_length) {
    CHECK_GE(new_length, 0);
    // we invoke resize() when new_length >= column_len
    if (column_len <= new_length) {
      X.resize(new_length, 0.0);
      idx.resize(new_length, 0);
      if (if_has_field) {
        field.resize(new_length, 0);
      }
    }
    column_len = new_length;
  }

  // Copy data from one SparseRow to another.
  void CopyFrom(SparseRow* row) {
    CHECK_NOTNULL(row);
    if_has_field = row->if_has_field;
    Resize(row->column_len);
    std::copy(row->X.begin(), row->X.end(),
              this->X.begin());
    std::copy(row->idx.begin(), row->idx.end(),
              this->idx.begin());
    if (if_has_field) {
      std::copy(row->field.begin(), row->field.end(),
                this->field.begin());
    }
  }

  std::vector<real_t> X;       // Storing the feature value.
  std::vector<index_t> idx;    // Stroing the index of each feature.
  std::vector<index_t> field;  // Storing the field value.
  size_t column_len;           // Storing the size of current row.
  bool if_has_field;           // for ffm ?
};

//------------------------------------------------------------------------------
// DMatrix (data matrix) is used to store a batch of training dataset.
// For many large-scale ML problems, we cannot load all the training data
// into memory at once. So we can load a small batch of dataset into the
// DMatrix at each iteration.
//------------------------------------------------------------------------------
struct DMatrix {
  // Constructor and Destructor
  DMatrix() {  }
  ~DMatrix() { Release(); }

  explicit DMatrix(size_t length)
    : row(length, nullptr),
      Y(length, 0.0),
      row_len(length),
      // we cannot release DMatrix before intialization
      can_release(false) {  }

  // Reset current row length
  void Setlength(size_t new_length) {
    CHECK_GE(new_length, 0);
    row_len = new_length;
  }

  // Resize current DMatrix
  void Resize(size_t new_length) {
    CHECK_GE(new_length, 0);
    Release();
    row_len = new_length;
    row.resize(new_length, nullptr);
    Y.resize(new_length, 0.0);
    row_len = new_length;
    can_release = false;
  }

  // Initialize the row pointers. On default the field is empty.
  void InitSparseRow(bool has_field = false) {
    // Pointers have been initialized
    if (can_release) {
      LOG(FATAL) << "Attempt to initialize the SparseRow twice.";
    }
    this->has_field = has_field;
    for (size_t i = 0; i < row_len; ++i) {
      if (row[i] == nullptr) {
        row[i] = new SparseRow(0, has_field);
      }
    }
    can_release = true;
  }

  // Copy data from one buffer to another. Note that we assume that
  // the row_len of original matrix >= that of the new matrix.
  void CopyFrom(DMatrix& matrix) {
    CHECK_GE(row_len, matrix.row_len);
    Resize(matrix.row_len);
    InitSparseRow(matrix.has_field);
    for (size_t i = 0; i < row_len; ++i) {
      Y[i] = matrix.Y[i];
      row[i]->CopyFrom(matrix.row[i]);
    }
  }

  // Release memory of all SparseRows.
  void Release() {
    // To avoid double free
    if (can_release) {
      STLDeleteElementsAndClear(&row);
    }
  }

  // Serialize matrix to disk file
  void Serialize(const std::string& filename) {
    CHECK_NE(filename.empty(), true);
    FILE* file = OpenFileOrDie(filename.c_str(), "w");
    // has_field, row_len, can_release
    WriteDataToDisk(file, (char*)(&has_field), sizeof(has_field));
    WriteDataToDisk(file, (char*)(&row_len), sizeof(row_len));
    WriteDataToDisk(file, (char*)(&can_release), sizeof(can_release));
    // Y
    WriteDataToDisk(file, (char*)Y.data(), sizeof(real_t)*Y.size());
    // row
    for (int i = 0; i < row.size(); ++i) {
      int col = row[i]->column_len;
      WriteDataToDisk(file, (char*)(&row[i]->column_len), sizeof(size_t));
      WriteDataToDisk(file, (char*)(&row[i]->if_has_field), sizeof(bool));
      WriteDataToDisk(file, (char*)row[i]->X.data(), sizeof(real_t)*col);
      WriteDataToDisk(file, (char*)row[i]->idx.data(), sizeof(index_t)*col);
      if (has_field) {
        WriteDataToDisk(file, (char*)row[i]->field.data(), sizeof(index_t)*col);
      }
    }
    Close(file);
  }

  // Deserialize matrix from disk file
  void Deserialize(const std::string& filename) {
    CHECK_NE(filename.empty(), true);
    FILE* file = OpenFileOrDie(filename.c_str(), "r");
    // has_field, row_len, can_release
    ReadDataFromDisk(file, (char*)(&has_field), sizeof(has_field));
    ReadDataFromDisk(file, (char*)(&row_len), sizeof(row_len));
    ReadDataFromDisk(file, (char*)(&can_release), sizeof(can_release));
    // Init row
    this->Resize(row_len);
    this->InitSparseRow(has_field);
    // Y
    ReadDataFromDisk(file, (char*)Y.data(), sizeof(real_t)*Y.size());
    // row
    for (int i = 0; i < row.size(); ++i) {
      ReadDataFromDisk(file, (char*)(&row[i]->column_len), sizeof(size_t));
      ReadDataFromDisk(file, (char*)(&row[i]->if_has_field), sizeof(bool));
      // Init row
      int col = row[i]->column_len;
      row[i]->Resize(row[i]->column_len);
      ReadDataFromDisk(file, (char*)row[i]->X.data(), sizeof(real_t)*col);
      ReadDataFromDisk(file, (char*)row[i]->idx.data(), sizeof(index_t)*col);
      if (has_field) {
        ReadDataFromDisk(file,
           (char*)row[i]->field.data(),
           sizeof(index_t)*col);
      }
    }
    Close(file);
  }

  // Storing SparseRows. Note that we use pointers here in order to
  // implement zero copy when copying data betweent different DMatrix(s).
  std::vector<SparseRow*> row;
  // Y can be either -1 or 0 (for negetive examples), and
  // can be 1 (for positive examples).
  std::vector<real_t> Y;
  // for ffm ?
  bool has_field;
  // Row length of current DMatrix.
  size_t row_len;
  // To avoid double free.
  bool can_release;
};

} // namespace xLearn

#endif // XLEARN_DATA_DATA_STRUCTURE_H_
