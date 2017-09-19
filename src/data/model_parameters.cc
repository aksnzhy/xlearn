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

This file is the implementation of the Model class.
*/

#include "src/data/model_parameters.h"

#include "src/base/common.h"
#include "src/base/file_util.h"
#include "src/base/math.h"
#include "src/base/scoped_ptr.h"

namespace xLearn {

//------------------------------------------------------------------------------
// The Model class
//------------------------------------------------------------------------------

// Basic contributor.
void Model::Initialize(const std::string& score_func,
                  const std::string& loss_func,
                  index_t num_feature,
                  index_t num_field,
                  index_t num_K) {
  CHECK(!score_func.empty());
  CHECK(!loss_func.empty());
  CHECK_GT(num_feature, 0);
  CHECK_GE(num_field, 0);
  CHECK_GE(num_K, 0);
  if (score_func == "linear") {
    param_num_w_ = num_feature;
  } else if (score_func == "fm") {
    param_num_w_ = num_feature +
                   num_feature * num_K;
  } else if (score_func == "ffm") {
    param_num_w_ = num_feature +
            num_feature * num_K * num_field;
  } else {
    LOG(FATAL) << "Unknow score function: " << score_func;
  }
  score_func_ = score_func;
  loss_func_ = loss_func;
  num_feat_ = num_feature;
  num_field_ = num_field;
  num_K_ = num_K;
  Initialize_w();
}

// To get the best performance for SSE and AVX, we need
// to allocate memory for the model parameters in aligned way
// For AVX, the align number should be 32, and for SSE, the
// align number should be 16
void Model::Initialize_w(bool set_value) {
  try {
    // Only used in Unix-like systems
  #ifdef __AVX__
    posix_memalign((void**)&param_w_, 32,
       param_num_w_ * sizeof(real_t));
  #else // SSE
    posix_memalign((void**)&param_w_, 16,
       param_num_w_ * sizeof(real_t));
  #endif
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for current  \
                   model parameters. Parameter size: [w] "
               << param_num_w_;
  }
  if (set_value) {
    // Set linear term to zero
    for (index_t i = 0; i < num_feat_; ++i) {
      param_w_[i] = 0.0;
    }
    // Init latent factor using RandDistribution()
    real_t coef = 1.0f / sqrt(num_K_);
    RandDistribution(param_w_ + num_feat_,
        param_num_w_ - num_feat_,
        0.0, 1.0, coef);
  }
}

// Initialize model from a checkpoint file
Model::Model(const std::string& filename) {
  CHECK_NE(filename.empty(), true);
  if (this->Deserialize(filename) == false) {
    printf("Cannot Load model from the file: %s\n",
           filename.c_str());
    exit(0);
  }
}

// Serialize current model to a disk file
void Model::Serialize(const std::string& filename) {
  CHECK_NE(filename.empty(), true);
  FILE* file = OpenFileOrDie(filename.c_str(), "w");
  // Write score function
  WriteStringToFile(file, score_func_);
  // Write loss function
  WriteStringToFile(file, loss_func_);
  // Write feature num
  WriteDataToDisk(file, (char*)&num_feat_, sizeof(num_feat_));
  // Write field num
  WriteDataToDisk(file, (char*)&num_field_, sizeof(num_field_));
  // Write K
  WriteDataToDisk(file, (char*)&num_K_, sizeof(num_K_));
  // Write w
  this->serialize_w(file);
  Close(file);
}

// Deserialize model from a checkpoint file.
bool Model::Deserialize(const std::string& filename) {
  CHECK_NE(filename.empty(), true);
  FILE* file = OpenFileOrDie(filename.c_str(), "r");
  if (file == NULL) { return false; }
  // Read score function
  ReadStringFromFile(file, score_func_);
  // Read loss function
  ReadStringFromFile(file, loss_func_);
  // Read feature num
  ReadDataFromDisk(file, (char*)&num_feat_, sizeof(num_feat_));
  // Read field num
  ReadDataFromDisk(file, (char*)&num_field_, sizeof(num_field_));
  // Read K
  ReadDataFromDisk(file, (char*)&num_K_, sizeof(num_K_));
  // Read w
  this->deserialize_w(file);
  Close(file);
  return true;
}

// Serialize w
void Model::serialize_w(FILE* file) {
  // Write size
  WriteDataToDisk(file, (char*)&param_num_w_, sizeof(param_num_w_));
  // Write data
  WriteDataToDisk(file, (char*)param_w_, sizeof(real_t)*param_num_w_);
}

// Deserialize w and v
void Model::deserialize_w(FILE* file) {
  // Read size
  ReadDataFromDisk(file, (char*)&param_num_w_, sizeof(param_num_w_));
  // Allocate memory
  Initialize_w(false);  /* do not set value */
  // Read data
  ReadDataFromDisk(file, (char*)param_w_, sizeof(real_t)*param_num_w_);
}

}  // namespace xLearn
