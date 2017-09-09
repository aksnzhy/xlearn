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

// Hyper parameters for Gaussian distribution.
static const real_t kInitMean = 0.0;
static const real_t kInitStdev = 0.1;

// Basic contributor.
void Model::Initialize(index_t num_param,
                       const std::string& score_func,
                       const std::string& loss_func,
                       index_t num_feature,
                       int num_field,
                       int num_K,
                       bool gaussian) {
  CHECK_GT(num_param, 0);
  CHECK_NE(score_func.empty(), true);
  CHECK_NE(loss_func.empty(), true);
  CHECK_GT(num_feature, 0);
  CHECK_GE(num_field, 0);
  CHECK_GE(num_K, 0);
  parameters_num_ = num_param;
  score_func_ = score_func;
  loss_func_ = loss_func;
  num_feat_ = num_feature;
  num_field_ = num_field;
  num_K_ = num_K;
  try {
    parameters_.resize(parameters_num_, 0.0);
    if (gaussian) {
      InitModelUsingGaussian();
    }
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for current  \
                   model parameters. Parameter size: "
               << parameters_num_;
  }
}

// Initialize model from a checkpoint file.
Model::Model(const std::string& filename) {
  CHECK_NE(filename.empty(), true);
  if (this->LoadModel(filename) == false) {
    printf("Cannot Load model from the file: %s\n",
           filename.c_str());
    exit(0);
  }
}

// Serialize current model to a checkpoint file.
void Model::SaveModel(const std::string& filename) {
  static std::string data_line;
  CHECK_NE(filename.empty(), true);
  FILE* file = OpenFileOrDie(StringPrintf("%s",
                        filename.c_str()).c_str(), "w");
  // The 1st line: score function
  // The 2nd line: loss function
  // The 3nd line: feature num
  data_line = StringPrintf("%s\n%s\n%d\n",
                    score_func_.c_str(),
                    loss_func_.c_str(),
                    num_feat_);
  // The 4th line: number of K (used in fm and ffm)
  if (score_func_.compare("fm") == 0 ||
      score_func_.compare("ffm") == 0) {
    data_line = StringPrintf("%s%d\n",
                      data_line.c_str(), num_K_);
  }
  // The 5th line: number of field (used in ffm)
  if (score_func_.compare("ffm") == 0) {
    data_line = StringPrintf("%s%d\n",
                      data_line.c_str(), num_field_);
  }
  WriteDataToDisk(file, data_line.c_str(), data_line.size());
  // Then, write param
  WriteVectorToFile<real_t>(file, this->parameters_);
  Close(file);
}

// Deserialize model from a checkpoint file.
bool Model::LoadModel(const std::string& filename) {
  static std::string data_line;
  CHECK_NE(filename.empty(), true);
  FILE* file = OpenFileOrDie(StringPrintf("%s",
                        filename.c_str()).c_str(), "r");
  if (file == NULL) { return false; }
  // The 1st line: score function
  GetLine(file, score_func_);
  // The 2nd line: loss function
  GetLine(file, loss_func_);
  // The 3nd line: feature num
  GetLine(file, data_line);
  num_feat_ = atoi(data_line.c_str());
  data_line.clear();
  // The 4nd line: number of K (used in fm and ffm)
  if (score_func_.compare("fm") == 0 ||
      score_func_.compare("ffm") == 0) {
    GetLine(file, data_line);
    num_K_ = atoi(data_line.c_str());
    data_line.clear();
  }
  // The 5th line: number of field (used in ffm)
  if (score_func_.compare("ffm") == 0) {
    GetLine(file, data_line);
    num_field_ = atoi(data_line.c_str());
    data_line.clear();
  }
  // Load param
  ReadVectorFromFile<real_t>(file, this->parameters_);
  parameters_num_ = parameters_.size();
  Close(file);
  return true;
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

} // namespace xLearn
