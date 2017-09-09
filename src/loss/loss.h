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
This file defines the Loss class, which is also called error
function or objective function.
*/

#ifndef XLEARN_LOSS_LOSS_H_
#define XLEARN_LOSS_LOSS_H_

#include <vector>

#include "src/base/common.h"
#include "src/base/class_register.h"
#include "src/base/math.h"
#include "src/data/model_parameters.h"
#include "src/updater/updater.h"
#include "src/score/score_function.h"

namespace xLearn {

//------------------------------------------------------------------------------
// The Loss is an abstract class, which can be implemented by the real
// loss functions such as cross-entropy loss (cross_entropy_loss.h),
// squared loss (squared_loss.h), hinge loss (hinge_loss.h), etc.
// There are three important method in Loss, including Evalute(), Predict(),
// and CalcGrad(). We can use the Loss class like this:
//
//   // Create a AbsLoss with linear score function, which
//   // is usually used for linear regression.
//   Loss* abs_loss = new AbsLoss();
//   abs_loss->Initialize(linear_score);
//
//   // Then, we can perform gradient descent like this:
//   DMatrix* matrix = NULL;
//   for (int n = 0: n < epoch; ++n) {
//     reader->Reset();
//     while (reader->Samples(matrix)) {
//       // Assume that the model and updater have been initialized
//       abs_loss->CalcGrad(matrix, model, updater);
//     }
//   }
//
//   // After training, we can calculate the train loss
//   real_t loss_val = 0;
//   index_t count = 0;
//   while (1) {
//     int tmp = reader->Samples(matrix);
//     if (tmp == 0) { break; }
//     pred.resize(tmp);
//     count += tmp;
//     abs_loss->Predict(matrix, model, pred);
//     loss_val += abs_loss->Evalute(pred, matrix->Y);
//   }
//   loss_val /= count;
//------------------------------------------------------------------------------
class Loss {
 public:
   // Constructor and Desstructor
   Loss() { };
   virtual ~Loss() { }

   // This function needs to be invoked before using this class.
   void Initialize(Score* score) { score_func_ = score; }

   // Given predictions and labels, return loss value.
   virtual real_t Evalute(const std::vector<real_t>& pred,
                          const std::vector<real_t>& label) = 0;

   // Given data sample and current model, return predictions.
   virtual void Predict(const DMatrix* data_matrix,
                        Model* model,
                        std::vector<real_t>& pred);

   // Given data sample and current model, calculate gradient
   // and update model.
   virtual void CalcGrad(const DMatrix* data_matrix,
                         Model* model,
                         Updater* updater) = 0;

   // The Sigmoid function, which map the predictions to 0~1.
   void Sigmoid(const std::vector<real_t>& pred,
                std::vector<real_t>& new_pred) {
     CHECK_EQ(pred.size(), new_pred.size());
     for (size_t i = 0; i < pred.size(); ++i) {
       new_pred[i] = fast_sigmoid(pred[i]);
     }
   }

 protected:
  // fast sigmoid function
  inline real_t fast_sigmoid(real_t x) {
    return 1.0f / (1.0f + fastexp(-x));
  }

  // The score function, e.g. LinearScore,
  // FMScore, FFMScore, etc.
  Score* score_func_;

 private:
  DISALLOW_COPY_AND_ASSIGN(Loss);
};

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_DEFINE_REGISTRY(xLearn_loss_registry, Loss);

#define REGISTER_LOSS(format_name, loss_name)               \
  CLASS_REGISTER_OBJECT_CREATOR(                            \
      xLearn_loss_registry,                                 \
      Loss,                                                 \
      format_name,                                          \
      loss_name)

#define CREATE_LOSS(format_name)                            \
  CLASS_REGISTER_CREATE_OBJECT(                             \
      xLearn_loss_registry,                                 \
      format_name)

} // namespace xLearn

#endif // XLEARN_LOSS_LOSS_H_
