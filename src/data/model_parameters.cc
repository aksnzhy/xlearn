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
#include "src/base/scoped_ptr.h"

namespace xLearn {

//------------------------------------------------------------------------------
// The Model class
//------------------------------------------------------------------------------

// Hyper parameters for Gaussian distribution.
static const real_t kInitMean = 0.0;
static const real_t kInitStdev = 0.01;
static const uint32 kMaxLineSize = 100 * 1024; // 100 KB for one line of data

// Basic contributor.
Model::Model(const HyperParam& hyper_param, bool gaussian) {
  parameters_num_ = hyper_param.num_param;
  score_func_ = hyper_param.score_func;
  num_feat_ = hyper_param.num_feature;
  num_field_ = hyper_param.num_field;
  num_K_ = hyper_param.num_K;
  CHECK_GE(parameters_num_, 0);
  CHECK_NE(score_func_.empty(), true);
  CHECK_GE(num_feat_, 0);
  CHECK_GE(num_field_, 0);
  CHECK_GE(num_K_, 0);
  try {
    parameters_.resize(parameters_num_, 0.0);
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
  static std::string data_line;
  CHECK_NE(filename.empty(), true);
  FILE* file = OpenFileOrDie(StringPrintf("%s_param",
                        filename.c_str()).c_str(), "w");
  // The 1st line: score function
  // The 2nd line: feature num
  data_line = StringPrintf("%s\n%d\n",
                    score_func_.c_str(), num_feat_);
  // The 3nd line: number of K (used in fm and ffm)
  if (score_func_.compare("fm") == 0 ||
      score_func_.compare("ffm") == 0) {
    data_line = StringPrintf("%s%d\n",
                      data_line.c_str(), num_K_);
  }
  // The 4th line: number of field (used in ffm)
  if (score_func_.compare("ffm") == 0) {
    data_line = StringPrintf("%s%d\n",
                      data_line.c_str(), num_field_);
  }
  WriteDataToDisk(file, data_line.c_str(), data_line.size());
  // Then, write param
  WriteVectorToFile<real_t>(file, this->parameters_);
  Close(file);
}

// Get one line of data from disk file
std::string Model::getline(FILE* file) {
  static scoped_array<char> line(new char[kMaxLineSize]);
  static std::string res_str;
  fgets(line.get(), kMaxLineSize, file);
  int read_len = strlen(line.get());
  if (line[read_len - 1] != '\n') {
    LOG(FATAL) << "Encountered a too-long line.   \
                   Please check the data.";
  } else {
    line[read_len - 1] = '\0';
    // Handle the txt format in DOS and windows.
    if (read_len > 1 && line[read_len - 2] == '\r') {
      line[read_len - 2] = '\0';
    }
  }
  res_str.assign(line.get());
  return res_str;
}

// Deserialize model from a checkpoint file.
void Model::LoadModel(const std::string& filename) {
  static std::string data_line;
  CHECK_NE(filename.empty(), true);
  FILE* file = OpenFileOrDie(StringPrintf("%s_param",
                        filename.c_str()).c_str(), "r");
  // The 1st line: score function
  score_func_ = getline(file);
  // The 2nd line: feature num
  data_line = getline(file);
  num_feat_ = atoi(data_line.c_str());
  data_line.clear();
  // The 3nd line: number of K (used in fm and ffm)
  if (score_func_.compare("fm") == 0 ||
      score_func_.compare("ffm") == 0) {
    data_line = getline(file);
    num_K_ = atoi(data_line.c_str());
    data_line.clear();
  }
  // The 4th line: number of field (used in ffm)
  if (score_func_.compare("ffm") == 0) {
    data_line = getline(file);
    num_field_ = atoi(data_line.c_str());
    data_line.clear();
  }
  // Load param
  ReadVectorFromFile<real_t>(file, this->parameters_);
  parameters_num_ = parameters_.size();
  Close(file);
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
