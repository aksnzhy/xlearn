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
#include "src/base/stringprintf.h"
#include "src/base/file_util.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"

namespace xLearn {

//------------------------------------------------------------------------------
// The Model class is responsible for storing global model prameters, which
// will be represented in a flat way, that is, no matter what model method we
// use, such as LR, FM, or FFM, we store the model parameters in a big array.
// We can make a checkpoint for current model, and we can also load a model
// checkpoint from disk file.
// A model can be initialized by Initialize() function or from a
// checkpoint file. We can use the Model class like this:
//
//    HyperParam hyper_param;
//    hyper_param.score_func = "ffm";
//    hyper_param.loss_func = "squared";
//    hyper_param.num_feature = 10;
//    hyper_param.num_K = 8;
//    hyper_param.num_field = 10;
//    hyper_param.num_param = kParameter_num;
//
//    Model model;
//    model_ffm.Initialize(hyper_param.num_param,
//                     hyper_param.score_func,
//                     hyper_param.loss_func,
//                     hyper_param.num_feature,
//                     hyper_param.num_field,
//                     hyper_param.num_K);
//
//    // Then, we can save model to a disk file.
//    model.SaveModel("/tmp/model.txt");
//
//    // We can get the model parameter vector:
//    vector<real_t>* param = model.GetParameter();
//
//    // Also, we can load model from this file.
//    Model new_model("/tmp/model.txt");
//------------------------------------------------------------------------------
class Model {
 public:
  // Default Constructor and Destructor
  Model() { }
  ~Model() { }

  // Initialize model from a checkpoint file.
  explicit Model(const std::string& filename);

  // Initialize model to zero or using
  // Gaussian distribution (by default)
  void Initialize(index_t num_param,
                  const std::string& score_func,
                  const std::string& loss_func,
                  index_t num_feature,
                  int num_field,
                  int num_K,
                  bool gaussian = true);

  // Serialize model to a checkpoint file.
  void SaveModel(const std::string& filename);

  // Deserialize model from a checkpoint file.
  bool LoadModel(const std::string& filename);

  // Get the pointer of current model parameters.
  std::vector<real_t>* GetParameter() { return &parameters_; }

  // Reset current model to init state.
  // We use the Gaussian distribution by default.
  void Reset(bool gaussion = true);

  // Save model parameters to a temp vector.
  void Saveweight(std::vector<real_t>& vec);

  // Load model parameters from a temp vector.
  void Loadweight(const std::vector<real_t>& vec);

  // Delete the model file and cache file.
  static void RemoveModelFile(const std::string& filename) {
    RemoveFile(StringPrintf("%s", filename.c_str()).c_str());
  }

  // Get functions
  size_t GetNumParameter() { return parameters_num_; }
  std::string GetScoreFunction() { return score_func_; }
  std::string GetLossFunction() { return loss_func_; }
  index_t GetNumFeature() { return num_feat_; }
  index_t GetNumField() { return num_field_; }
  int GetNumK() { return num_K_; }

 protected:
  std::vector<real_t> parameters_;       // Storing the model parameters.
  index_t             parameters_num_;   // Number of model parameters.
  std::string         score_func_;       // Score function
  std::string         loss_func_;        // Loss function
  index_t             num_feat_;         // Number of feature
  int                 num_field_;        // Number of field (used in ffm)
  int                 num_K_;            // Number of K (used in fm and ffm)

  // Initialize model using Gaussian distribution.
  void InitModelUsingGaussian();

 private:
  DISALLOW_COPY_AND_ASSIGN(Model);
};

} // namespace xLearn

#endif // XLEARN_DATA_MODEL_PARAMETERS_H_
