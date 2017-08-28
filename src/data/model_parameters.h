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

This file defines the class of model parameters
used by xLearn.
*/

#ifndef XLEARN_DATA_MODEL_PARAMETERS_H_
#define XLEARN_DATA_MODEL_PARAMETERS_H_

#include <vector>
#include <string>

#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"

namespace xLearn {

//------------------------------------------------------------------------------
// The Model class is responsible for storing global model prameters, which
// will be represented in a flat way, that is, no matter what model method we
// use, such as LR, FM, or FFM, we store the model parameters in a big array.
// We can make a checkpoint for current model, and we can also load a model
// checkpoint from disk file.
//------------------------------------------------------------------------------
class Model {
 public:
  // Default Constructor and Destructor
  Model() { }
  ~Model() { }

  // Set all parameters to 0 or using Gaussian distribution.
  explicit Model(const HyperParam& hyper_param,
                 bool gaussian = true);

  // Initialize model parameters from a checkpoint file.
  explicit Model(const std::string& filename);

  // Serialize model to a checkpoint file.
  void SaveModel(const std::string& filename);

  // Deserialize model from a checkpoint file.
  bool LoadModel(const std::string& filename);

  // Get the pointer of current model parameters.
  std::vector<real_t>* GetParameter() { return &parameters_; }

  // Get the length of current model parameters.
  index_t GetLength() { return parameters_num_; }

  // Reset current model to init state. We use the Gaussian
  // distribution by default.
  void Reset(bool gaussion = true);

  // Save model parameters to a temp vector.
  void Saveweight(std::vector<real_t>& vec);

  // Load model parameters from a temp vector.
  void Loadweight(const std::vector<real_t>& vec);

  // Delete the model file.
  void RemoveModelFile(const std::string filename);

  // Get function
  size_t GetNumParameter() { return parameters_num_; }
  std::string GetScoreFunction() { return score_func_; }
  index_t GetNumFeature() { return num_feat_; }
  index_t GetNumField() { return num_field_; }
  int GetNumK() { return num_K_; }

 protected:
  std::vector<real_t> parameters_;       // Storing the model parameters.
  size_t              parameters_num_;   // Number of model parameters.
  std::string         score_func_;       // linear, fm, or ffm
  index_t             num_feat_;         // Number of feature
  index_t             num_field_;        // Number of field (used in ffm)
  int                 num_K_;            // Number of K (used in fm and ffm)

  std::string getline(FILE* file_ptr);   // Get one line of data

  // Initialize model using Gaussian distribution.
  void InitModelUsingGaussian();

 private:
  DISALLOW_COPY_AND_ASSIGN(Model);
};

} // namespace xLearn

#endif // XLEARN_DATA_MODEL_PARAMETERS_H_
