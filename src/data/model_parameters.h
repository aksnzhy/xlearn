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
// will be represented in a flatten way, that is, no matter what model method
// we use, such as LR, FM, or FFM, we store the model parameters in a big
// array. We can make a checkpoint for current model, and we can also load
// a model checkpoint from disk file.
// A model can be initialized by Initialize() function or from a
// checkpoint file. We can use the Model class like this:
//
//    HyperParam hyper_param;
//    hyper_param.score_func = "ffm";
//    hyper_param.loss_func = "squared";
//    hyper_param.num_feature = 10;
//    hyper_param.num_K = 8;
//    hyper_param.num_field = 10;
//
//    Model model;
//    model_ffm.Initialize(hyper_param.score_func,
//                     hyper_param.loss_func,
//                     hyper_param.num_feature,
//                     hyper_param.num_field,
//                     hyper_param.num_K);
//
//    /* We can get the model parameter vector: */
//    vector<real_t>* param = model.GetParameter();
//
//    /* We can save model to a disk file: */
//    model.SaveModel("/tmp/model.txt");
//
//    /* Also, we can load model from this file. */
//    Model new_model("/tmp/model.txt");
//------------------------------------------------------------------------------
class Model {
 public:
  // Default Constructor and Destructor
  Model() { }
  ~Model() { }

  // Initialize model from a checkpoint file.
  explicit Model(const std::string& filename);

  // Initialize model parameters to zero or using
  // the Gaussian distribution
  void Initialize(const std::string& score_func,
              const std::string& loss_func,
              index_t num_feature,
              index_t num_field,
              index_t num_K);

  // Serialize model to a checkpoint file.
  void Serialize(const std::string& filename);

  // Deserialize model from a checkpoint file.
  bool Deserialize(const std::string& filename);

  // Get the pointer of current model parameters.
  std::vector<real_t>* GetParameter() { return &parameters_; }

  // Get functions
  size_t GetNumParameter() { return param_num_; }
  std::string GetScoreFunction() { return score_func_; }
  std::string GetLossFunction() { return loss_func_; }
  index_t GetNumFeature() { return num_feat_; }
  index_t GetNumField() { return num_field_; }
  index_t GetNumK() { return num_K_; }

 protected:
  /* Number of model parameters.
   For linear socre, param_num =  num_feat
   For fm, param_num = num_feat + num_feat * num_K
   For ffm, param_num = num_feat + num_feat * num_field * num_K */
  index_t  param_num_;
  /* Score function: 'linear', 'fm', or 'ffm' */
  std::string  score_func_;
  /* Loss function: 'squared', 'cross-entropy', etc */
  std::string  loss_func_;
  /* Number of feature (from 0, include bias) */
  index_t  num_feat_;
  /* Number of field (used in ffm, from 0) */
  index_t  num_field_;
  /* Number of K (used in fm and ffm) */
  index_t  num_K_;
  /* Storing the model parameters */
  std::vector<real_t> parameters_;

 private:
  DISALLOW_COPY_AND_ASSIGN(Model);
};

}  // namespace xLearn

#endif  // XLEARN_DATA_MODEL_PARAMETERS_H_
