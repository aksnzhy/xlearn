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
This file defines the Score class, including linear score,
FM score, FFM score, etc.
*/

#ifndef XLEARN_LOSS_SCORE_FUNCTION_H_
#define XLEARN_LOSS_SCORE_FUNCTION_H_

#include <vector>

#include "src/base/common.h"
#include "src/base/class_register.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"
#include "src/updater/updater.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Score is an abstract class, which can be implemented by different
// score functions such as LinearScore (liner_score.h), FMScore (fm_score.h)
// FFMScore (ffm_score.h) etc. On common, we initial a Score function and
// pass its pointer to a Loss class like this:
//
//  Score* score = new FMScore();
//  score->Initialize(num_feat, K, num_field);
//  Loss* loss = new AbsLoss();
//  loss->Initialize(score);
//
//  The CalcScore() and CalcGrad() function are used by Loss class.
//------------------------------------------------------------------------------
class Score {
 public:
  // Constructor and Desstructor
  Score() { }
  virtual ~Score() { }

  // This function needs to be invoked before using this class.
  virtual void Initialize(index_t num_feat, int k, int field) = 0;

  // Given one exmaple and current model, return the score.
  virtual real_t CalcScore(const SparseRow* row,
                           const std::vector<real_t>* w) = 0;

  // Calculate gradient and update current model parameters.
  virtual void CalcGrad(const SparseRow* row,
                        std::vector<real_t>& param,
                        real_t pg, /* partial gradient */
                        Updater* updater) = 0;

 private:
  index_t num_factor_;
  index_t num_feature_;
  index_t num_field_;

  DISALLOW_COPY_AND_ASSIGN(Score);
};

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_DEFINE_REGISTRY(xLearn_score_registry, Score);

#define REGISTER_SCORE(format_name, score_name)             \
  CLASS_REGISTER_OBJECT_CREATOR(                            \
      xLearn_score_registry,                                \
      Score,                                                \
      format_name,                                          \
      score_name)

#define CREATE_SCORE(format_name)                           \
  CLASS_REGISTER_CREATE_OBJECT(                             \
      xLearn_score_registry,                                \
      format_name)


} // namespace xLearn

#endif // XLEARN_LOSS_SCORE_FUNCTION_H_
