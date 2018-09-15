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
This file defines the basic data structures.
*/

#ifndef XLEARN_DATA_DATA_STRUCTURE_H_
#define XLEARN_DATA_DATA_STRUCTURE_H_

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

#include "src/base/common.h"
#include "src/base/file_util.h"
#include "src/base/stl-util.h"

namespace xLearn {

//------------------------------------------------------------------------------
// We use 32-bits float to store the real number during computation, 
// such as the model parameter and the gradient.
//------------------------------------------------------------------------------
typedef float real_t;

//------------------------------------------------------------------------------
// We use 32-bits unsigned integer to store the index 
// of the feature and the model parameters.
//------------------------------------------------------------------------------
typedef uint32 index_t;

//------------------------------------------------------------------------------
// Mapping sparse feature to dense feature. Used by distributed computation.
//------------------------------------------------------------------------------
typedef std::unordered_map<index_t, index_t> feature_map;

//------------------------------------------------------------------------------
// We use SSE to accelerate our training, and hence some 
// parameters will be aligned.
//------------------------------------------------------------------------------
const int kAlign = 4;
const int kAlignByte = 16;

//------------------------------------------------------------------------------
// MetricInfo stores the evaluation metric information, which
// will be printed for users during the training.
//------------------------------------------------------------------------------
struct MetricInfo {
  real_t loss_val;    /* Loss value */
  real_t metric_val;  /* Metric value */
};

//------------------------------------------------------------------------------
// Node is used to store information for each column of the feature vector.
// For tasks like LR and FM, we just need to store the feature id and the 
// feature value. While for tasks like FFM, we also need to store field id.
//------------------------------------------------------------------------------
struct Node {
  // Default constructor
  Node() { }
  Node(index_t field, index_t feat, real_t val)
   : field_id(field), 
     feat_id(feat), 
     feat_val(val) { }
  /* Field id is start from 0 */
  index_t field_id;
  /* Feature id is start from 0 */ 
  index_t feat_id;
  /* Feature value */ 
  real_t feat_val;
};

//------------------------------------------------------------------------------
// SparseRow is used to store one line of the data, which
// is represented as a vector of the Node data structure.
//------------------------------------------------------------------------------
typedef std::vector<Node> SparseRow;

//------------------------------------------------------------------------------
// DMatrix (data matrix) is used to store a batch of the dataset.
// It can be the whole dataset used in in-memory training, or just a
// working set used in on-disk training. This is because for many 
// large-scale machine learning problems, we cannot load all the data into 
// memory at once, and hence we have to load a small batch of dataset in 
// DMatrix at each samplling for training or prediction.  
// We can use the DMatrix like this:
//
//    DMatrix matrix;
//    /* Initialize 10 rows */
//    matrix.ResetMatrix(10);   
//    for (int i = 0; i < 10; ++i) {
//      matrix.Y[i] = ...
//      /* We set feild_id to 0 by default */
//      matrix.AddNode(i, feat_id, feat_val, field_id);
//    }
//
//    /* Serialize and Deserialize */
//    matrix.Serialize("/tmp/test.bin");
//    DMatrix new_matrix;
//    /* The new matrix is the same with old matrix */
//    new_matrix.Deserialize("/tmp/test.bin");
//
//    /* We can access the matrix */
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
//
//    /* We can also get the max index of feature or field */
//    index_t max_feat = matrix.MaxFeat();
//    index_t max_field = matrix.MaxField();
//------------------------------------------------------------------------------
struct DMatrix {
  // Constructor
  DMatrix()
   : hash_value_1(0), 
     hash_value_2(0),
     row_length(0),
     row(0),
     Y(0),
     norm(0),
     has_label(false),
     pos(0) { }

  // Destructor
  ~DMatrix() { }

  // Reset data for the DMatrix.
  // This function will first release the original
  // memory allocated for the DMatrix, and then re-allocate 
  // memory for this new matrix. For some dataset, it does not
  // contains the label y, and hence we need to set the 
  // has_label variable to false. On deafult, this value will
  // be set to true.
  void ResetMatrix(size_t length, bool label = true) {
    CHECK_GE(length, 0);
    this->Release();
    hash_value_1 = 0;
    hash_value_2 = 0;
    row_length = length;
    row.resize(length, nullptr);
    Y.resize(length, 0);
    // Here we set norm to 1.0 by default, which means
    // that we don't use instance-wise nomarlization
    norm.resize(length, 1.0);
    // Indicate that if current dataset has the label y
    has_label = label;
    pos = 0;
  }

  // Release memory for DMatrix.
  // Note that a typical alternative that forces a
  // reallocation is to use swap(), instead of using clear().
  void Release() {
    hash_value_1 = 0;
    hash_value_2 = 0;
    // Delete Y
    std::vector<real_t>().swap(Y);
    // Delete Node
    for (int i = 0; i < row_length; ++i) {
      if (row[i] != nullptr) {
        STLDeleteElementsAndClear(&row);
      }
    }
    // Delete SparseRow
    std::vector<SparseRow*>().swap(row);
    // Delete norm
    std::vector<real_t>().swap(norm);
    has_label = false;
    row_length = 0;
    pos = 0;
  }

  // Add node to current data matrix.
  // We don't use the 'field' by default because it
  // will only be used in the ffm tasks.
  void AddNode(index_t row_id,  
               index_t feat_id,
               real_t feat_val, 
               index_t field_id = 0) {
    CHECK_GT(row_length, row_id);
    // Allocate memory for the first adding
    if (row[row_id] == nullptr) {
      row[row_id] = new SparseRow;
    }
    Node node(field_id, feat_id, feat_val);
    row[row_id]->push_back(node);
  }

  // The hash value is used to identify the difference
  // between two data matrix, and it can be generated by HashFile() 
  // method (in file_util.h) and this value will be used when reading 
  // txt data from disk file. We can cache the binary data in disk file 
  // to accelerate the reading of disk file.
  void SetHash(uint64 hash_1, uint64 hash_2) {
    hash_value_1 = hash_1;
    hash_value_2 = hash_2;
  }

  // Copy another data matrix to this matrix.
  // Note that here we do the deep copy and we will
  // allocate memory if current matrix is empty.
  void CopyFrom(const DMatrix* matrix) {
    CHECK_NOTNULL(matrix);
    this->Release();
    // Copy hash value
    this->hash_value_1 = matrix->hash_value_1;
    this->hash_value_2 = matrix->hash_value_2;
    // Copy row length
    this->row_length = matrix->row_length;
    this->row.resize(row_length, nullptr);
    // Copy row
    for (index_t i = 0; i < row_length; ++i) {
      SparseRow* rowc = matrix->row[i];
      for (SparseRow::iterator iter = rowc->begin();
           iter != rowc->end(); ++iter) {
        this->AddNode(i, 
                iter->feat_id, 
                iter->feat_val, 
                iter->field_id);
      }
    }
    // Copy y
    this->Y = matrix->Y;
    // Copy norm
    this->norm = matrix->norm;
    // Copy has label
    this->has_label = matrix->has_label;
    // Copy pos
    this->pos = matrix->pos;
  }

  // Compress current sparse matrix to a dense matrix.
  // This method will be used in distributed computation.
  // For example, the sparse matrix is:
  //  ------------------------------------------
  //  |    1:0.1    5:0.1    8:0.1   10:0.1    |
  //  |    3:0.1    12:0.1   20:0.1            |
  //  |    5:0.1    8:0.1    11:0.1            |
  //  |    2:0.1    4:0.1    7:0.1             |
  //  ------------------------------------------
  // After compress, we can get a dense matrix like this:
  //  ------------------------------------------
  //  |    1:0.1    5:0.1    7:0.1    8:0.1    |
  //  |    3:0.1    10:0.1   11:0.1            |
  //  |    5:0.1    7:0.1    9:0.1             |
  //  |    2:0.1    4:0.1    6:0.1             |
  //  ------------------------------------------
  // Also, we can get a vector to store the mapping relations:
  //  -------------------------------------------------
  //  | 1 | 2 | 3 | 4 | 5 | 7 | 8 | 10 | 11 | 12 | 20 |
  //  -------------------------------------------------
  void Compress(std::vector<index_t>& feature_list) {
    // Using a map to store the mapping relations
    size_t node_num {0};
    for (auto row : this->row) {
      node_num += row->size();
    }
    std::unordered_set<index_t> feat_set;
    feat_set.reserve(node_num);
    for (index_t i = 0; i < this->row_length; ++i) {
      SparseRow* row = this->row[i];
      for (SparseRow::iterator iter = row->begin();
           iter != row->end(); ++iter) {
        if (feat_set.count(iter->feat_id) == 0) {
          feat_set.insert(iter->feat_id);
        }
      }
    }
    feature_list.reserve(feat_set.size());
    std::copy(feat_set.begin(), feat_set.end(), 
              std::back_inserter(feature_list));
    std::sort(begin(feature_list), end(feature_list));
    feature_map mp;
    mp.reserve(feature_list.size());
    for (index_t i = 0; i < feature_list.size(); ++ i) {
      mp[feature_list[i]] = i + 1;
    }
    for (index_t i = 0; i < this->row_length; ++ i) {
      for (auto &iter: *this->row[i]) {
        // using map is better than lower_bound
        iter.feat_id = mp[iter.feat_id];
      }
    }
  }

  // Get a mini-batch of data from curremt data matrix. 
  // This method will be used for distributed computation. 
  // Return the count of sample for each function call.
  index_t GetMiniBatch(index_t batch_size, DMatrix& mini_batch) {
    CHECK_EQ(batch_size, mini_batch.row_length);
    // Copy mini-batch
    for (index_t i = 0; i < batch_size; ++i) {
      if (this->pos >= this->row_length) {
        return i;
      }
      mini_batch.row[i] = this->row[pos];
      mini_batch.Y[i] = this->Y[pos];
      mini_batch.norm[i] = this->norm[pos];
      this->pos++;
    }
    return batch_size;
  }

  // Serialize current DMatrix to disk file.
  void Serialize(const std::string& filename) {
    CHECK_NE(filename.empty(), true);
    CHECK_EQ(row_length, row.size());
    CHECK_EQ(row_length, Y.size());
    CHECK_EQ(row_length, norm.size());
    FILE* file = OpenFileOrDie(filename.c_str(), "w");
    // Write hash_value
    WriteDataToDisk(file, (char*)&hash_value_1, sizeof(hash_value_1));
    WriteDataToDisk(file, (char*)&hash_value_2, sizeof(hash_value_2));
    // Write row_length
    WriteDataToDisk(file, (char*)&row_length, sizeof(row_length));
    // Write row
    for (size_t i = 0; i < row_length; ++i) {
      WriteVectorToFile(file, *(row[i]));
    }
    // Write Y
    WriteVectorToFile(file, Y);
    // Write norm
    WriteVectorToFile(file, norm);
    // Write has_label
    WriteDataToDisk(file, (char*)&has_label, sizeof(has_label));
    // Write pos
    WriteDataToDisk(file, (char*)&pos, sizeof(pos));
    Close(file);
  }

  // Deserialize the DMatrix from disk file.
  void Deserialize(const std::string& filename) {
    CHECK(!filename.empty());
    this->Release();
    FILE* file = OpenFileOrDie(filename.c_str(), "r");
    // Read hash_value
    ReadDataFromDisk(file, (char*)&hash_value_1, sizeof(hash_value_1));
    ReadDataFromDisk(file, (char*)&hash_value_2, sizeof(hash_value_2));
    // Read row_length
    ReadDataFromDisk(file, (char*)&row_length, sizeof(row_length));
    CHECK_GE(row_length, 0);
    // Read row
    row.resize(row_length, nullptr);
    for (size_t i = 0; i < row_length; ++i) {
      row[i] = new SparseRow;
      ReadVectorFromFile(file, *(row[i]));
    }
    // Read Y
    ReadVectorFromFile(file, Y);
    // Read norm
    ReadVectorFromFile(file, norm);
    // Read has label
    ReadDataFromDisk(file, (char*)&has_label, sizeof(has_label));
    // Read pos
    ReadDataFromDisk(file, (char*)&pos, sizeof(pos));
    Close(file);
  }

  // We get find the max index of feature or field in current
  // data matrix. This is used for initialize our model parameter.  
  inline index_t MaxFeat() const { return max_feat_or_field(true); }
  inline index_t MaxField() const { return max_feat_or_field(false); }
  inline index_t max_feat_or_field(bool is_feat) const {
    index_t max = 0;
    for (size_t i = 0; i < row_length; ++i) {
      SparseRow* sr = this->row[i];
      for (SparseRow::const_iterator iter = sr->begin();
           iter != sr->end(); ++iter) {
        if (is_feat) {  // feature
          if (iter->feat_id > max) {
            max = iter->feat_id;
          }
        } else {  // field
          if (iter->field_id > max) {
            max = iter->field_id;
          }
        }
      }
    }
    return max;
  }

  /* The DMatrix has a hash value that is geneerated 
  from the TXT file. These two values are used to check 
  whether we can use binary file to speedup data reading */
  uint64 hash_value_1;
  uint64 hash_value_2;
  /* Row length of current matrix */
  index_t row_length;
  /* Store many SparseRow. Using pointer for zero-copy */
  std::vector<SparseRow*> row;
  /* (0 or -1) for negative and (+1) for positive
  examples, and others value for regression */
  std::vector<real_t> Y;
  /* Used for instance-wise normalization */
  std::vector<real_t> norm;
  /* If current dataset has label y */
  bool has_label;
  /* Current position for GetMiniBatch() */
  index_t pos;
};

}  // namespace xLearn

#endif  // XLEARN_DATA_DATA_STRUCTURE_H_
