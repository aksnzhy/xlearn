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
This file defines the Metric class, which can be used for
Accuracy, Precision, Recall, ROC, AUC, MAE, MSE, etc.
*/

#ifndef XLEARN_LOSS_METRIC_H_
#define XLEARN_LOSS_METRIC_H_

#include "src/base/common.h"
#include "src/data/data_structure.h"

namespace xLearn {

class Metric {
 public:
  Metric() { }
  ~Metric() { }

  // Call this function before we use this class
  void Initialize(const std::string& metric) {
    if (metric.compare("acc") != 0 &&     // Accuracy
        metric.compare("prec") != 0 &&    // Precision
        metric.compare("recall") != 0 &&
        metric.compare("f1") != 0 &&
        metric.compare("mae") != 0 &&
        metric.compare("mape") != 0) {
      LOG(FATAL) << "Unknow metric: " << metric;
    }
    metric_type_ = metric;
  }

  // Get metric type
  std::string type() {
    if (metric_type_.compare("acc") == 0) {
      return "accuracy";
    } else if (metric_type_.compare("prec") == 0) {
      return "precision";
    } else if (metric_type_.compare("recall") == 0) {
      return "recall";
    } else if (metric_type_.compare("f1") == 0) {
      return "F1";
    } else if (metric_type_.compare("mae") == 0) {
      return "MAP";
    } else if (metric_type_.compare("mape") == 0) {
      return "MAPE";
    }
    LOG(ERROR) << "Unknow metric: " << metric_type_;
    return 0;
  }

  // Will be used during training
  void Accumulate(const std::vector<real_t>& Y,
                  const std::vector<real_t>& pred) {

  }

  // Return metric value
  std::string GetMetric() {
    return "00.00";
  }

 private:
  std::string metric_type_;
  index_t real_pos_example_;
  index_t real_neg_example_;
  index_t pre_pos_example_;
  index_t pre_neg_example_;
  // A set of metric funtions
  real_t Accuracy();
  real_t Precision();
  real_t Recall();
  real_t F1();
  real_t MAE();
  real_t MAPE();

  DISALLOW_COPY_AND_ASSIGN(Metric);
};

}  // namespace xLearn

#endif  // XLEARN_LOSS_METRIC_H_
