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
        metric.compare("roc") != 0 &&
        metric.compare("auc") != 0 &&
        metric.compare("mae") != 0 &&
        metric.compare("mse") != 0) {
      LOG(FATAL) << "Unknow metric: " << metric;
    }
    metric_type_ = metric;
  }

  // Get metric type
  std::string type() {
    if (metric_type_.compare("acc") == 0) {
      return "Accuracy";
    } else if (metric_type_.compare("prec") == 0) {
      return "Precision";
    } else if (metric_type_.compare("recall") == 0) {
      return "Recall";
    } else if (metric_type_.compare("roc") == 0) {
      return "ROC";
    } else if (metric_type_.compare("auc") == 0) {
      return "AUC";
    } else if (metric_type_.compare("mae") == 0) {
      return "MAE";
    } else if (metric_type_.compare("mse") == 0) {
      return "MSE";
    }
    LOG(ERROR) << "Unknow metric: " << metric_type_;
    return 0;
  }

  // Set value
  void Set(index_t real_pos_example,  // number of positive example
      index_t real_neg_example,  // number of negative example
      index_t pre_pos_example,  // right prediction for positive example
      index_t pre_neg_example) {  // right prediction for negative exmaple
    CHECK_GE(real_pos_example, 0);
    CHECK_GE(real_neg_example, 0);
    CHECK_GE(pre_pos_example, 0);
    CHECK_GE(pre_neg_example, 0);
    real_pos_example_ = real_pos_example;
    real_neg_example_ = real_neg_example;
    pre_pos_example_ = pre_pos_example;
    pre_neg_example_ = pre_neg_example;
  }

  // Will be used during training
  void Accumulate(index_t* real_pos_example,
                  index_t* real_neg_example,
                  index_t* pre_pos_example,
                  index_t* pre_neg_example,
                  const std::vector<real_t>& Y,
                  const std::vector<real_t>& pred) {
    for (index_t i = 0; i < pred.size(); ++i) {
      if (Y[i] > 0) {
        (*real_pos_example)++;
        if (pred[i] >= 0) {
          (*pre_pos_example)++;
        }
      } else {
        (*real_neg_example)++;
        if (pred[i] < 0) {
          (*pre_neg_example)++;
        }
      }
    }
  }

  // Return metric value
  real_t GetMetric() {
    if (metric_type_.compare("acc") == 0) {
      return Accuracy();
    } else if (metric_type_.compare("prec") == 0) {
      return Precision();
    } else if (metric_type_.compare("recall") == 0) {
      return Recall();
    } else if (metric_type_.compare("roc") == 0) {
      return ROC();
    } else if (metric_type_.compare("auc") == 0) {
      return AUC();
    } else if (metric_type_.compare("mae") == 0) {
      return MAE();
    } else if (metric_type_.compare("mse") == 0) {
      return MSE();
    }
    LOG(ERROR) << "Unknow metric: " << metric_type_;
    return 0;
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
  real_t ROC();
  real_t AUC();
  real_t MAE();
  real_t MSE();

  DISALLOW_COPY_AND_ASSIGN(Metric);
};

}  // namespace xLearn

#endif  // XLEARN_LOSS_METRIC_H_
