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
This file defines the class of model parameters.
*/

#ifndef XLEARN_DATA_MODEL_PARAMETERS_H_
#define XLEARN_DATA_MODEL_PARAMETERS_H_

#include <string>

#include <math.h>

#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/base/logging.h"

namespace xLearn {

//------------------------------------------------------------------------------
// The Model class is responsible for storing the global
// model prameters. We can dump a checkpoint for current model
// and we can also load a model checkpoint from disk file.
// A model can be initialized by Initialize() function or from a
// checkpoint file through constructor.
// We can use the Model class like this:
//
//    HyperParam hyper_param;
//    hyper_param.score_func = "ffm";
//    hyper_param.loss_func = "squared";
//    hyper_param.num_feature = 10;
//    hyper_param.num_K = 8;  
//    hyper_param.num_field = 10;
//    hyper_param.model_scale = 0.66;
//    hyper_param.auxiliary_size = 2;
//
//    Model model;
//    model_ffm.Initialize(hyper_param.score_func,
//                     hyper_param.loss_func,
//                     hyper_param.num_feature,
//                     hyper_param.num_field,
//                     hyper_param.num_K,
//                     hyper_param.auxiliary_size,
//                     hyper_param.model_scale);
//
//    /* We can get the parameter of the linear term: */
//    real_t* w = model.GetParameter_w();
//    index_t w_len = model.GetNumParameter_w();
//    for (index_t i = 0; i < w_len; ++i) {
//      /* access w[i] ... */
//    }
//
//    /* We can also get the parameter of the latent factor */
//    real_t* v = model.GetParameter_v();
//    index_t v_len = model.GetNumParameter_v();
//    for (index_t i = 0; i < v_len; ++i) {
//      /* access v[i] ... */
//    }
//
//    /* We can save model to a disk file: */
//    model.SaveModel("/tmp/model.txt");
//
//    /* Also, we can load model from this file. */
//    Model new_model("/tmp/model.txt");
//
// The Model class can support early-stopping technique. We can set
// a record for the best model parameter by using SetBestModel() and
// we can shrink back to find the best model by using Shrink() method.
//------------------------------------------------------------------------------
class Model {
 public:
  // Default Constructor and Destructor
  Model() { }
  ~Model() { free_model(); }

  // Initialize model from a checkpoint file.
  explicit Model(const std::string& filename);

  // Initialize model parameters to zero or using
  // a random distribution.
  void Initialize(const std::string& score_func,
              const std::string& loss_func,
              index_t num_feature,
              index_t num_field,
              index_t num_K,
              index_t aux_size,
              real_t scale = 1.0);

  // Serialize model to a checkpoint file.
  void Serialize(const std::string& filename);

  // Serialize model to a TXT file.
  void SerializeToTXT(const std::string& filename);

  // Deserialize model from a checkpoint file.
  bool Deserialize(const std::string& filename);

  // Take a record of the best model during training.
  void SetBestModel();

  // Shrink back for getting the best model.
  void Shrink();

  // Get the size of auxiliary cache size.
  inline real_t GetAuxiliarySize() { return aux_size_; }

  // Get the pointer of linear term.
  inline real_t* GetParameter_w() { return param_w_; }

  // Get the pointer of latent factor.
  inline real_t* GetParameter_v() { return param_v_; }

  // Get the pointer of bias.
  inline real_t* GetParameter_b() { return param_b_; }

  // Get the size of the linear term.
  inline index_t GetNumParameter_w() { return param_num_w_; }

  // Get the size of the latent factor.
  // For linear score this value equals zero.
  inline index_t GetNumParameter_v() { return param_num_v_; }

  // Reset current model parameters.
  inline void Reset() { set_value(); }

  // Get score function type.
  inline std::string& GetScoreFunction() { return score_func_; }

  // Get the loss function type.
  inline std::string& GetLossFunction() { return loss_func_; }

  // Get the number of feature.
  inline index_t GetNumFeature() { return num_feat_; }

  // Get the number of field.
  inline index_t GetNumField() { return num_field_; }

  // Get the number of k.
  inline index_t GetNumK() { return num_K_; }

  // Get the aligned size of K.
  inline index_t get_aligned_k() {
    return (index_t)ceil((real_t)num_K_/kAlign)*kAlign;
  }

  // Get the total size of model parameters.
  // 2 = bias + bias_gradient
  inline index_t GetNumParameter() {
    return param_num_w_ + param_num_v_ + 2;
  }

 protected:
  /* Score function
  For now it can be 'linear', 'fm', or 'ffm' */
  std::string  score_func_;
  /* Loss function
  For now it can be 'squared' and 'cross-entropy' */
  std::string  loss_func_;
  /* Size of the linear term
  Note that we store both of the model parameters
  and the gradient cache in param_w_, so
  param_num_w_ = num_feat_ * 2  */
  index_t param_num_w_;
  /* Size of the latent factor. 
  We store both the model parameters and the gradient 
  cache for adagrad in param_v_. 
  For linear function, param_num_v = 0
  For fm function, param_num_v_ = num_feat * num_K * 2
  For ffm funtion, param_num_v_ = num_feat * num_field * num_K * 2  */
  index_t  param_num_v_;
  /* Number of feature
  Feature id is start from 0 */
  index_t  num_feat_;
  /* Number of field (Used in ffm)
  Field id is start from 0 */
  index_t  num_field_;
  /* Number of K (used in fm and ffm)
  Becasue we use SSE, so the real k should be aligned.
  User can get the aligned K by using get_aligned_k() */
  index_t  num_K_;
  /* Auxiliary memory size for different optimization method
  For 'adagrad' it equals 2 and 'ftrl' it equals 3 */
  index_t aux_size_;
  /* Storing the parameter of linear term */
  real_t*  param_w_ = nullptr;
  /* Storing the parameter of latent factor */
  real_t*  param_v_ = nullptr;
  /* Storing the bias term */
  real_t*  param_b_ = nullptr;
  /* The following varibles are used for early-stopping */
  real_t* param_best_w_ = nullptr;
  real_t* param_best_v_ = nullptr;
  real_t* param_best_b_ = nullptr;
  /* Used to init model parameters */
  real_t scale_;

  // Initialize the value of model parameters and gradient cache.
  void initial(bool set_value = false);

  // Reset the value of current model parameters.
  void set_value();

  // Serialize w, v, b to disk file.
  void serialize_w_v_b(FILE* file);

  // Deserialize w, v, b from disk file.
  void deserialize_w_v_b(FILE* file);

  // Free the allocated memory.
  void free_model();

 private:
  DISALLOW_COPY_AND_ASSIGN(Model);
};

}  // namespace xLearn

#endif  // XLEARN_DATA_MODEL_PARAMETERS_H_
