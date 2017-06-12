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
#include "src/base/stringprintf.h"

namespace xLearn {

//------------------------------------------------------------------------------
// The Model class
//------------------------------------------------------------------------------

// Hyper parameters for Gaussian distribution.
static const real_t kInitMean = 0.0;
static const real_t kInitStdev = 0.01;

// Basic contributor.
Model::Model(size_t parameter_num, size_t cache_num, bool gaussian) :
  parameters_num_(parameter_num), cache_num_(cache_num) {
  CHECK_GE(parameters_num_, 0);
  CHECK_GE(cache_num_, 0);
  try {
    parameters_.resize(parameters_num_, 0.0);
    param_cache_.resize(cache_num_, 0.0);
    if (gaussian) {
      InitModelUsingGaussian();
    }
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for current      \
                   model parameters. Parameter size: "
               << parameters_num_;
  }
}

// Initialize model from a checkpoint file.
Model::Model(const std::string& filename) {
  CHECK_NE(filename.empty(), true);
  this->LoadModel(filename);
}

// Serialize current model to a checkpoint file.
void Model::SaveModel(const std::string& filename) {
  CHECK_NE(filename.empty(), true);
  FILE* file_ptr_param =
      OpenFileOrDie(StringPrintf("%s_param", filename.c_str()).c_str(), "w");
  // Write param
  WriteVectorToFile<real_t>(file_ptr_param, this->parameters_);
  // Write cache
  WriteVectorToFile<real_t>(file_ptr_param, this->param_cache_);
  Close(file_ptr_param);
}

// Deserialize model from a checkpoint file.
void Model::LoadModel(const std::string& filename) {
  CHECK_NE(filename.empty(), true);
  FILE* file_ptr_param =
      OpenFileOrDie(StringPrintf("%s_param", filename.c_str()).c_str(), "r");
  // Load param
  ReadVectorFromFile<real_t>(file_ptr_param, this->parameters_);
  parameters_num_ = parameters_.size();
  // Load cache
  ReadVectorFromFile<real_t>(file_ptr_param, this->param_cache_);
  cache_num_ = param_cache_.size();
  Close(file_ptr_param);
}

// Reset current model to init state.
void Model::Reset(bool gaussian) {
  if (gaussian) {
    InitModelUsingGaussian();
  } else {
    for (size_t i = 0; i < parameters_num_; ++i) {
      parameters_[i] = 0.0;
    }
  }
}

// Save model parameters to a tmp vector
void Model::Saveweight(std::vector<real_t>& vec) {
  CHECK_EQ(parameters_num_, vec.size());
  copy(parameters_.begin(), parameters_.end(), vec.begin());
}

// Load model parameters from a temp vector
void Model::Loadweight(const std::vector<real_t>& vec) {
  CHECK_EQ(parameters_num_, vec.size());
  copy(vec.begin(), vec.end(), parameters_.begin());
}

// Initialize model parameters using Gaussian distribution.
void Model::InitModelUsingGaussian() {
  CHECK_EQ(parameters_num_, parameters_.size());
  for (size_t i = 0; i < parameters_num_; ++i) {
    parameters_[i] = ran_gaussion(kInitMean, kInitStdev);
  }
}

// Delete the model file and cache file.
void Model::RemoveModelFile(const std::string filename) {
  // Remove model file
  RemoveFile(StringPrintf("%s_param",
                          filename.c_str()).c_str());
}

} // namespace xLearn
