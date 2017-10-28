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
This file is the implementation of the Metric class.
*/

#include "src/loss/metric.h"

namespace xLearn {

real_t Metric::Accuracy() const {
  real_t res = 0;
  res = (true_pos_ * 1.0 + true_neg_) / counter_;
  return res;
}

real_t Metric::Precision() const {
  real_t res = 0;
  res = (true_pos_ * 1.0) / (true_pos_ + false_pos_);
  return res;
}

real_t Metric::Recall() const {
  real_t res = 0;
  res = (true_pos_ * 1.0) / (true_pos_ + false_neg_);
  return res;
}

real_t Metric::F1() const {
  real_t res = 0;
  res = (2.0 * true_pos_) / (counter_ + true_pos_ - true_neg_);
  return res;
}

real_t Metric::AUC() const {
  return 0;
}

real_t Metric::MAE() const {
  return error_accum_ * 1.0 / counter_;
}

real_t Metric::MAPE() const {
  return error_accum_ * 1.0 / counter_;;
}

}  // namespace xLearn
