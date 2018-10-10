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
This file is the implementation of the Trainer class.
*/

#include <stdio.h>
#include <vector>
#include <string>

#include "src/solver/trainer.h"
#include "src/data/data_structure.h"
#include "src/base/timer.h"

namespace xLearn {

/*********************************************************
 *  Show head info                                       *
 *********************************************************/
void Trainer::show_head_info(bool validate) {
  std::vector<std::string> str_list;
  std::vector<int> width_list;
  str_list.push_back("Epoch");
  width_list.push_back(6);
  str_list.push_back("Train " + loss_->loss_type());
  width_list.push_back(20);
  if (validate) {
    str_list.push_back("Test " + loss_->loss_type());
    width_list.push_back(20);
    if (metric_ != nullptr) {
      str_list.push_back("Test " + metric_->metric_type());
      width_list.push_back(20);
    }
  }
  str_list.push_back("Time cost (sec)");
  width_list.push_back(20);
  Color::Modifier green(Color::FG_GREEN);
  Color::Modifier reset(Color::RESET);
  std::cout << green << "[------------]" << reset;
  Color::print_row(str_list, width_list);
}

/*********************************************************
 *  Show train info                                      *
 *********************************************************/
void Trainer::show_train_info(real_t tr_loss, 
                              real_t te_loss,
                              real_t te_metric,
                              real_t time_cost, 
                              bool validate,
                              index_t epoch) {
  std::vector<std::string> str_list;
  std::vector<int> width_list;
  str_list.push_back(StringPrintf("%d", epoch));
  width_list.push_back(6);
  str_list.push_back(StringPrintf("%.6f", tr_loss));
  width_list.push_back(20);
  if (validate) {
    str_list.push_back(StringPrintf("%.6f", te_loss));
    width_list.push_back(20);
    if (metric_ != nullptr) {
      str_list.push_back(StringPrintf("%.6f", te_metric));
      width_list.push_back(20);
    }
  }
  str_list.push_back(StringPrintf("%.2f", time_cost));
  width_list.push_back(20);
  Color::Modifier green(Color::FG_GREEN);
  Color::Modifier reset(Color::RESET);
  std::cout << green << "[ " << reset;
  std::cout.width(4); 
  std::cout << static_cast<int>(epoch*1.0/epoch_*100) 
            << "%" << green << "      ]"  << reset;
  Color::print_row(str_list, width_list);
}

/*********************************************************
 *  Basic train                                          *
 *********************************************************/
void Trainer::Train() {
  // Get train Reader and test Reader
  std::vector<Reader*> tr_reader;
  tr_reader.push_back(reader_list_[0]);
  std::vector<Reader*> te_reader;
  if (reader_list_.size() == 2) {
    te_reader.push_back(reader_list_[1]);
  }
  this->train(tr_reader, te_reader);
}

/*********************************************************
 *  Cross-Validation                                     *
 *********************************************************/
void Trainer::CVTrain() {
  // Use the i-th reader as validation Reader
  for (int i = 0; i < reader_list_.size(); ++i) {
    Color::print_action(
      StringPrintf("Cross-validation: %d/%lu:", 
        i+1, reader_list_.size())
    );
    // Get the train Reader and test Reader
    std::vector<Reader*> tr_reader;
    for (int j = 0; j < reader_list_.size(); ++j) {
      if (i == j) { continue; }
      tr_reader.push_back(reader_list_[j]);
    }
    std::vector<Reader*> te_reader;
    te_reader.push_back(reader_list_[i]);
    if (i != 0) {
      // Re-init current model parameters.
      model_->Reset();
    }
    this->train(tr_reader, te_reader);
  }
  // Average metric for cross-validation
  show_average_metric();
}

/*********************************************************
 *  Calc average evaluation metric for CV                *
 *********************************************************/
void Trainer::show_average_metric() {
  real_t loss = 0;
  real_t metric = 0;
  for (size_t i = 0; i < metric_info_.size(); ++i) {
    loss += metric_info_[i].loss_val;
    if (metric_ != nullptr) {
      metric += metric_info_[i].metric_val;
    }
  }
  Color::print_info(
    StringPrintf("Average %s: %.6f", 
    loss_->loss_type().c_str(), 
    loss / metric_info_.size())
  );
  if (metric_ != nullptr) {
    Color::print_info(
      StringPrintf("Average %s: %.6f", 
      metric_->metric_type().c_str(),
       metric / metric_info_.size())
    );
  }
}

/*********************************************************
 *  Basic train function                                 *
 *********************************************************/
void Trainer::train(std::vector<Reader*>& train_reader,
                    std::vector<Reader*>& test_reader) {
  int best_epoch = 0;
  int stop_window = 0;
  real_t best_result = metric_ == nullptr ? kFloatMax : kFloatMin;
  real_t prev_result = metric_ == nullptr ? kFloatMin : kFloatMax;
  MetricInfo te_info;
  // Show header info
  if (!quiet_) { 
    show_head_info(!test_reader.empty()); 
  }
  for (int n = 1; n <= epoch_; ++n) {
    Timer timer;
    timer.tic();
    // Calc grad and update model
    real_t tr_loss = calc_gradient(train_reader);
    // we don't do any evaluation in a quiet model
    if (!quiet_) {
      if (!test_reader.empty()) { 
        te_info = calc_metric(test_reader); 
      }
      // show evaludation metric info
      show_train_info(tr_loss, 
                      te_info.loss_val,
                      te_info.metric_val,
                      timer.toc(), 
                      !test_reader.empty(), 
                      n);
      // Early-stopping
      if (early_stop_) {
        if ((metric_ == nullptr && te_info.loss_val <= best_result) ||
            (metric_ != nullptr && te_info.metric_val >= best_result))  {
          best_result = metric_ == nullptr ? 
            te_info.loss_val : te_info.metric_val;
          best_epoch = n;
          model_->SetBestModel();
        }
        if ((metric_ == nullptr && te_info.loss_val > prev_result) ||
            (metric_ != nullptr && te_info.metric_val < prev_result)) {
          // If the validation loss goes up conntinuously
          // in stop_window epoch, we stop training
          if (stop_window == stop_window_) { break; }
          stop_window++;
        } else {
          stop_window = 0;
        }
        prev_result = metric_ == nullptr ? 
          te_info.loss_val : te_info.metric_val;
      }
    }
  }
  if (early_stop_ && best_epoch != epoch_) {  // not for cv
    std::string metric_name = metric_ == nullptr ? 
      "loss" : metric_->metric_type();
    Color::print_action(
      StringPrintf("Early-stopping at epoch %d, best %s: %f", 
        best_epoch, metric_name.c_str(), best_result)
    );
    model_->Shrink();
  } else {  // for cv
    metric_info_.push_back(te_info);
  }
}

/*********************************************************
 *  Calc gradient and update model                       *
 *********************************************************/
real_t Trainer::calc_gradient(std::vector<Reader*>& reader) {
  CHECK_NE(reader.empty(), true);
  loss_->Reset();
  for (int i = 0; i < reader.size(); ++i) {
    reader[i]->Reset();
    DMatrix* matrix = nullptr;
    for (;;) {
      index_t tmp = reader[i]->Samples(matrix);
      if (tmp == 0) { break; }
      loss_->CalcGrad(matrix, *model_);
    }
  }
  return loss_->GetLoss();
}

/*********************************************************
 *  Calc evaluation metric                               *
 *********************************************************/
MetricInfo Trainer::calc_metric(std::vector<Reader*>& reader_list) {
  CHECK_NE(reader_list.empty(), true);
  DMatrix* matrix = nullptr;
  std::vector<real_t> pred;
  if (metric_ != nullptr) {
    metric_->Reset();
  }
  loss_->Reset();
  for (int i = 0; i < reader_list.size(); ++i) {
    reader_list[i]->Reset();
    for (;;) {
      index_t tmp = reader_list[i]->Samples(matrix);
      if (tmp == 0) { break; }
      if (tmp != pred.size()) { pred.resize(tmp); }
      loss_->Predict(matrix, *model_, pred);
      loss_->Evalute(pred, matrix->Y);
      if (metric_ != nullptr) {
        metric_->Accumulate(matrix->Y, pred);
      }
    }
  }
  MetricInfo info;
  info.loss_val = loss_->GetLoss();
  if (metric_ != nullptr) {
    info.metric_val = metric_->GetMetric();
  }
  return info;
}

} // namespace xLearn
