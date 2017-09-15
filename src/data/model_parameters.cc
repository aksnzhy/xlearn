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
    param_num_ = num_feature;
  } else if (score_func == "fm") {
    param_num_ = num_feature + num_feature * num_K;
  } else if (score_func == "ffm") {
    param_num_ = num_feature + num_feature * num_K * num_field;
  } else {
    LOG(FATAL) << "Unknow score function: " << score_func;
  }
  score_func_ = score_func;
  loss_func_ = loss_func;
  num_feat_ = num_feature;
  num_field_ = num_field;
  num_K_ = num_K;
  try {
    parameters_.resize(param_num_, 0.0);
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for current  \
                   model parameters. Parameter size: "
               << param_num_;
  }
  // Initialize model using random Gaussian distribution
  if (score_func_ == "fm" || score_func_ == "ffm") {
    real_t coef = 1.0f / sqrt(num_K_);
    RandDistribution(this->parameters_, 0.0, 1.0, coef);
    for (index_t i = 0; i < num_feat_; ++i) {
      // Reset 0.0 for linear term
      this->parameters_[i] = 0.0;
    }
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
  // Write parameter num
  WriteDataToDisk(file, (char*)&param_num_, sizeof(param_num_));
  // Write parameter data
  WriteVectorToFile(file, this->parameters_);
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
  // Read param_num_
  ReadDataFromDisk(file, (char*)&param_num_, sizeof(param_num_));
  // Read parameter data
  ReadVectorFromFile(file, this->parameters_);
  Close(file);
  return true;
}

}  // namespace xLearn
