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

double Metric::Accuracy() {
  double res = 0;
  res = (pre_pos_example_ * 1.0 + pre_neg_example_) /
        (real_pos_example_ + real_neg_example_);
  return res;
}

double Metric::Precision() {
  return 0;
}

double Metric::Recall() {
  return 0;
}

double Metric::ROC() {
  return 0;
}

double Metric::AUC() {
  return 0;
}

double Metric::MAE() {
  return 0;
}

double Metric::MSE() {
  return 0;
}

}  // namespace xLearn
