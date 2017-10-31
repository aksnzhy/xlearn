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

  // Call this function before we use the Metric class
  void Initialize(const std::string& metric) {
    if (metric.compare("acc") != 0 &&     // Accuracy
        metric.compare("prec") != 0 &&    // Precision
        metric.compare("recall") != 0 &&
        metric.compare("f1") != 0 &&
        metric.compare("auc") != 0 &&
        metric.compare("mae") != 0 &&
        metric.compare("mape") != 0 &&
        metric.compare("rmsd") != 0 &&
        metric.compare("none") != 0) {
      LOG(FATAL) << "Unknow metric: " << metric;
    }
    metric_type_ = metric;
    counter_ = 0;
    true_pos_ = 0;
    false_pos_ = 0;
    true_neg_ = 0;
    false_neg_ = 0;
    error_accum_ = 0.0;
  }

  // Get metric type
  std::string type() const {
    if (metric_type_.compare("acc") == 0) {
      return "accuracy";
    } else if (metric_type_.compare("prec") == 0) {
      return "precision";
    } else if (metric_type_.compare("recall") == 0) {
      return "recall";
    } else if (metric_type_.compare("f1") == 0) {
      return "F1";
    } else if (metric_type_.compare("auc") == 0) {
      return "AUC";
    } else if (metric_type_.compare("mae") == 0) {
      return "MAP";
    } else if (metric_type_.compare("mape") == 0) {
      return "MAPE";
    } else if (metric_type_.compare("rmsd") == 0) {
      return "RMSD";
    } else if (metric_type_.compare("none") == 0) {
      return "NONE";
    }
    LOG(ERROR) << "Unknow metric: " << metric_type_;
    return 0;
  }

  // Accumulate counters during the training
  void Accumulate(const std::vector<real_t>& Y,
                  const std::vector<real_t>& pred) {
    CHECK_NE(pred.empty(), true);
    CHECK_NE(Y.empty(), true);
    counter_ += pred.size();
    for (size_t i = 0; i < pred.size(); ++i) {
      if (metric_type_.compare("mae") == 0) {
        error_accum_ += (abs(Y[i] - pred[i]));
      } else if (metric_type_.compare("mape") == 0) {
        error_accum_ += (abs(Y[i] - pred[i]) / Y[i]);
      } else if (metric_type_.compare("rmsd") == 0) {
        real_t tmp = abs(Y[i] - pred[i]);
        error_accum_ += (tmp*tmp);
      } else {
        if (pred[i] >= 0) {  // for positive prediction
          if (Y[i] == 1) {
            ++true_pos_;
          } else {
            ++false_pos_;
          }
        } else {  // for negative prediction
          if (Y[i] == 1) {
            ++false_neg_;
          } else {
            ++true_neg_;
          }
        }
      }
    }
  }

  // Reset counters for the next epoch
  void Reset() {
    counter_ = 0;
    true_pos_ = 0;
    false_pos_ = 0;
    true_neg_ = 0;
    false_neg_ = 0;
    error_accum_ = 0.0;
  }

  // Return metric value
  real_t GetMetric() const {
    if (metric_type_.compare("acc") == 0) {
      return Accuracy();
    } else if (metric_type_.compare("prec") == 0) {
      return Precision();
    } else if (metric_type_.compare("recall") == 0) {
      return Recall();
    } else if (metric_type_.compare("f1") == 0) {
      return F1();
    } else if (metric_type_.compare("AUC") == 0) {
      return AUC();
    } else if (metric_type_.compare("mae") == 0) {
      return MAE();
    } else if (metric_type_.compare("mape") == 0) {
      return MAPE();
    } else if (metric_type_.compare("rmsd") == 0) {
      return RMSD();
    }
    LOG(ERROR) << "Unknow metric: " << metric_type_;
    return 0;
  }

protected:
  /* Can be 'acc', 'prec', 'recall', 'f1'
     'mae', and 'mape' */
  std::string metric_type_;
  /* The number of total example */
  index_t counter_;
  /* The number of true positive */
  index_t true_pos_;
  /* The number of false positive */
  index_t false_pos_;
  /* The number of true negative */
  index_t true_neg_;
  /* The number of false negative */
  index_t false_neg_;
  /* Sum of error for regression tasks */
  real_t error_accum_;
  // A set of metric funtions
  real_t Accuracy() const;
  real_t Precision() const;
  real_t Recall() const;
  real_t F1() const;
  real_t AUC() const;
  real_t MAE() const;
  real_t MAPE() const;
  real_t RMSD() const;

  // Return the absolute value
  inline real_t abs(real_t a) { return a >= 0 ? a : -a; }

 private:
  DISALLOW_COPY_AND_ASSIGN(Metric);
};

}  // namespace xLearn

#endif  // XLEARN_LOSS_METRIC_H_
