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
This file defines the LinearScore class.
*/

#ifndef XLEARN_DIST_LINEAR_SCORE_H_
#define XLEARN_DIST_LINEAR_SCORE_H_

#include "src/base/common.h"
#include "src/data/model_parameters.h"
#include "src/distributed/dist_score_function.h"

namespace xLearn {

//------------------------------------------------------------------------------
// LinearScore is used to implemente generalized linear
// models (GLMs), where the socre function is y = wTx.
//------------------------------------------------------------------------------
class DistLinearScore : public DistScore {
 public:
  // Constructor and Desstructor
  DistLinearScore() { }
  ~DistLinearScore() { }

  // Given one exmaple and current model, this method
  // returns the linear score wTx.
  real_t CalcScore(const SparseRow* row,
                   std::unordered_map<index_t, real_t>* w,
                   real_t norm = 1.0);

  void DistCalcGrad(const DMatrix* matrix,
                     std::unordered_map<index_t, real_t>* w,
                     real_t* sum,
                     std::unordered_map<index_t, real_t>* g,
                     index_t start_idx,
                     index_t end_idx);

 protected:
  // Calculate gradient and update model using sgd
  void calc_grad_sgd(const DMatrix* matrix,
                     std::unordered_map<index_t, real_t>* w,
                     real_t* sum,
                     std::unordered_map<index_t, real_t>* g,
                     real_t start_idx,
                     real_t end_idx
                    );

  // Calculate gradient and update model using adagrad
  void calc_grad_adagrad(const DMatrix* matrix,
                         std::unordered_map<index_t, real_t>* w,
                         real_t* sum,
                         std::unordered_map<index_t, real_t>* g,
                         real_t start_idx,
                         real_t end_idx
                        );

  // Calculate gradient and update model using ftrl
  void calc_grad_ftrl(const DMatrix* matrix,
                      std::unordered_map<index_t, real_t>* w,
                      real_t* sum,
                      std::unordered_map<index_t, real_t>* g,
                      real_t start_idx,
                      real_t end_idx
                     );

 private:
  DISALLOW_COPY_AND_ASSIGN(DistLinearScore);
};

}  // namespace xLearn

#endif  // XLEARN_LINEAR_SCORE_H_
