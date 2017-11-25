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
This file defines the FFMScore (field-aware factorization machine) class.
*/

#ifndef XLEARN_LOSS_FFM_SCORE_FTRL_H_
#define XLEARN_LOSS_FFM_SCORE_FTRL_H_

#include "src/base/common.h"
#include "src/score/score_function.h"

namespace xLearn {

//------------------------------------------------------------------------------
// FFMScore is used to implemente field-aware factorization machines,
// in which the socre function is:
//   y = sum( (V_i_fj*V_j_fi)(x_i * x_j) )
// Here leave out the bias and linear term.
//------------------------------------------------------------------------------
class FFMScoreFtrl : public Score {
public:
 // Constructor and Desstructor
 FFMScoreFtrl() {
   int ret = posix_memalign(
       (void**)&comp_res1,
       kAlignByte,
       1 * sizeof(real_t));
   ret = posix_memalign(
       (void**)&comp_res2,
       kAlignByte,
       1 * sizeof(real_t));

   ret = posix_memalign(
       (void**)&comp_z_lt_zero,
       kAlignByte,
       1 * sizeof(real_t));
   ret = posix_memalign(
       (void**)&comp_z_gt_zero,
       kAlignByte,
       1 * sizeof(real_t));
 }
 ~FFMScoreFtrl() { }

 // Given one exmaple and current model, this method
 // returns the ffm score.
 real_t CalcScore(const SparseRow* row,
                  Model& model,
                  real_t norm = 1.0);

 // Calculate gradient and update current
 // model parameters.
 void CalcGrad(const SparseRow* row,
               Model& model,
               real_t pg,
               real_t norm = 1.0);

 private:
  real_t* comp_res1 = nullptr;
  real_t* comp_res2 = nullptr;
  real_t* comp_z_lt_zero = nullptr;
  real_t* comp_z_gt_zero = nullptr;

 private:
  DISALLOW_COPY_AND_ASSIGN(FFMScoreFtrl);
};

}  // namespace xLearn

#endif  // XLEARN_LOSS_FFM_SCORE_FTRL_H_
