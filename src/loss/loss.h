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
This file defines the Loss class, which is also called error
function or objective function.
*/

#ifndef XLEARN_LOSS_LOSS_H_
#define XLEARN_LOSS_LOSS_H_

#include <vector>
#include <string>

#include "src/base/common.h"
#include "src/base/class_register.h"
#include "src/base/math.h"
#include "src/base/thread_pool.h"
#include "src/data/model_parameters.h"
#include "src/score/score_function.h"

namespace xLearn {

//------------------------------------------------------------------------------
// The Loss is an abstract class, which can be implemented by the real
// loss functions such as cross-entropy loss (cross_entropy_loss.h),
// squared loss (squared_loss.h), hinge loss (hinge_loss.h), etc.
// There are three important method in Loss, including Evalute(), Predict(),
// and CalcGrad(). We can use the Loss class like this:
//
//   // Create a squared loss with linear score function, which
//   // is usually used for linear regression.
//   Loss* sq_loss = new SquaredLoss();
//   sq_loss->Initialize(linear_score, pool);
//
//   // Then, we can perform gradient descent like this:
//   DMatrix* matrix = NULL;
//   for (int n = 0: n < epoch; ++n) {
//     reader->Reset();
//     while (reader->Samples(matrix)) {
//       // Assume that the model and updater have been initialized
//       sq_loss->CalcGrad(matrix, model, updater);
//     }
//   }
//
//   // After training, we can calculate the train loss
//   real_t loss_val = 0;
//   while (1) {
//     int tmp = reader->Samples(matrix);
//     if (tmp == 0) { break; }
//     pred.resize(tmp);
//     sq_loss->Predict(matrix, model, pred);
//     sq_loss->Evalute(pred, matrix->Y);
//   }
//   loss_val = sq_loss->GetLoss()
//------------------------------------------------------------------------------
class Loss {
 public:
  // Constructor and Desstructor
  Loss() : loss_sum_(0), total_example_ (0) { };
  virtual ~Loss() { }

  // This function needs to be invoked before using this class
  void Initialize(Score* score, 
                  ThreadPool* pool, 
                  bool norm = true,
                  bool lock_free = false,
                  index_t batch_size = 0) {
    CHECK_NOTNULL(score);
    CHECK_NOTNULL(pool);
    CHECK_GE(batch_size, 0);
    score_func_ = score;
    pool_ = pool;
    norm_ = norm;
    threadNumber_ = pool_->ThreadNumber();
    lock_free_ = lock_free;
    batch_size_ = batch_size;
  }

  // Given predictions and labels, accumulate loss value.
  virtual void Evalute(const std::vector<real_t>& pred,
                       const std::vector<real_t>& label) = 0;

  // Given data sample and current model, return predictions.
  virtual void Predict(const DMatrix* data_matrix,
                       Model& model,
                       std::vector<real_t>& pred);

  // Given data sample and current model, calculate gradient
  // and update current model parameters.
  // This function will also acummulate loss value.
  virtual void CalcGrad(const DMatrix* data_matrix, 
                        Model& model) = 0;

  // Given data sample and current model, calculate gradient.
  // Note that this method doesn't update local model, and the
  // gradient will be pushed to the parameter server, which is 
  // used for distributed computation.
  virtual void CalcGradDist(DMatrix* data_matrix,
                            Model& model,
                            std::vector<real_t>& grad);

  // Return the calculated loss value
  virtual real_t GetLoss() {
    return loss_sum_ / total_example_;
  }

  // Reset loss_sum_ and total_example_
  virtual void Reset() {
    loss_sum_ = 0;
    total_example_ = 0;
  }

  // Return a current loss type
  virtual std::string loss_type() = 0;

 protected:
  /* The score function, including LinearScore,
  FMScore, FFMScore, etc */
  Score* score_func_;
  /* Use instance-wise normalization */
  bool norm_;
  /* Open lock-free training ? */
  bool lock_free_;
  /* Thread pool for multi-thread training */
  ThreadPool* pool_;
  /* Number of thread in thread pool */
  size_t threadNumber_;
  /* Used to store the accumulate loss */
  real_t loss_sum_;
  /* Used to store the number of example */
  index_t total_example_;
  /* Mini-batch size */
  index_t batch_size_;

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

}  // namespace xLearn

#endif  // XLEARN_LOSS_LOSS_H_
