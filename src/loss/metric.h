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
Accuracy, Precision, Recall, F1, MAE, MAPE, etc.
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
  real_t GetMetric() {
    return 0;
  }

protected:
  /* Can be 'acc', 'prec', 'recall', 'f1'
     'mae', and 'mape' */
  std::string metric_type_;
  /* The number of real positive */
  index_t real_pos_;
  /* The number of real negative */
  index_t real_neg_;
  /* The number of true positive */
  index_t true_pos_;
  /* The number of false positive */
  index_t false_pos_;
  /* The number of true negative */
  index_t true_neg_;
  /* The number of false negative */
  index_t false_neg_;
  // A set of metric funtions
  real_t Accuracy();
  real_t Precision();
  real_t Recall();
  real_t F1();
  real_t MAE();
  real_t MAPE();

 private:
  DISALLOW_COPY_AND_ASSIGN(Metric);
};

}  // namespace xLearn

#endif  // XLEARN_LOSS_METRIC_H_
