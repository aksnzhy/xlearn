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
This file tests the Metric class.
*/

#include "gtest/gtest.h"

#include <vector>

#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/loss/metric.h"

namespace xLearn {

TEST(AccMetricTest, acc_test) {
  std::vector<real_t> Y;
  Y.push_back(1.0);
  Y.push_back(1.0);
  Y.push_back(-1.0);
  Y.push_back(-1.0);
  std::vector<real_t> pred;
  pred.push_back(100);
  pred.push_back(32);
  pred.push_back(12);
  pred.push_back(-21);
  AccMetric metric;
  size_t threadNumber = std::thread::hardware_concurrency();
  ThreadPool* pool = new ThreadPool(threadNumber);
  metric.Initialize(pool);
  metric.Accumulate(Y, pred);
  real_t metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, (3.0 / 4.0));
  metric.Reset();
  Y[0] = 1.0;
  Y[1] = -1.0;
  Y[2] = -1.0;
  Y[3] = 1.0;
  pred[0] = 12;
  pred[1] = 12;
  pred[2] = 12;
  pred[3] = -12;
  metric.Accumulate(Y, pred);
  metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, (1.0 / 4.0));
  EXPECT_EQ(metric.metric_type(), "Accuarcy");
}

TEST(PrecMetricTest, prec_test) {
  std::vector<real_t> Y;
  Y.push_back(1.0);
  Y.push_back(1.0);
  Y.push_back(-1.0);
  Y.push_back(-1.0);
  std::vector<real_t> pred;
  pred.push_back(100);
  pred.push_back(32);
  pred.push_back(12);
  pred.push_back(-21);
  PrecMetric metric;
  size_t threadNumber = std::thread::hardware_concurrency();
  ThreadPool* pool = new ThreadPool(threadNumber);
  metric.Initialize(pool);
  metric.Accumulate(Y, pred);
  real_t metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, (2.0 / 3.0));
  metric.Reset();
  Y[0] = 1.0;
  Y[1] = -1.0;
  Y[2] = -1.0;
  Y[3] = -1.0;
  pred[0] = 12;
  pred[1] = 12;
  pred[2] = 12;
  pred[3] = -12;
  metric.Accumulate(Y, pred);
  metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, (1.0 / 3.0));
  EXPECT_EQ(metric.metric_type(), "Precision");
}

TEST(RecallMetricTest, recall_test) {
  std::vector<real_t> Y;
  Y.push_back(1.0);
  Y.push_back(1.0);
  Y.push_back(-1.0);
  Y.push_back(-1.0);
  std::vector<real_t> pred;
  pred.push_back(100);
  pred.push_back(32);
  pred.push_back(12);
  pred.push_back(-21);
  RecallMetric metric;
  size_t threadNumber = std::thread::hardware_concurrency();
  ThreadPool* pool = new ThreadPool(threadNumber);
  metric.Initialize(pool);
  metric.Accumulate(Y, pred);
  real_t metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, (2.0 / 2.0));
  metric.Reset();
  Y[0] = 1.0;
  Y[1] = -1.0;
  Y[2] = 1.0;
  Y[3] = -1.0;
  pred[0] = 12;
  pred[1] = 12;
  pred[2] = -12;
  pred[3] = -12;
  metric.Accumulate(Y, pred);
  metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, (1.0 / 2.0));
  EXPECT_EQ(metric.metric_type(), "Recall");
}

TEST(F1MetricTest, f1_test) {
  std::vector<real_t> Y;
  Y.push_back(1.0);
  Y.push_back(1.0);
  Y.push_back(-1.0);
  Y.push_back(-1.0);
  std::vector<real_t> pred;
  pred.push_back(100);
  pred.push_back(32);
  pred.push_back(12);
  pred.push_back(-21);
  F1Metric metric;
  size_t threadNumber = std::thread::hardware_concurrency();
  ThreadPool* pool = new ThreadPool(threadNumber);
  metric.Initialize(pool);
  metric.Accumulate(Y, pred);
  real_t metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, (4.0 / 5.0));
  metric.Reset();
  Y[0] = 1.0;
  Y[1] = -1.0;
  Y[2] = 1.0;
  Y[3] = -1.0;
  pred[0] = 12;
  pred[1] = 12;
  pred[2] = -12;
  pred[3] = -12;
  metric.Accumulate(Y, pred);
  metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, (2.0 / 4.0));
  EXPECT_EQ(metric.metric_type(), "F1");
}

TEST(AUCMetricTest, auc_test) {
  std::vector<real_t> Y = {-1.0, -1.0, 1.0, 1.0};
  std::vector<real_t> pred = {0.1, 0.4, 0.35, 0.8};
  AUCMetric metric;
  size_t threadNumber = std::thread::hardware_concurrency();
  ThreadPool* pool = new ThreadPool(threadNumber);
  metric.Initialize(pool);
  metric.Accumulate(Y, pred);
  real_t metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, 0.75);
  metric.Reset();
  Y = {-1.0, -1.0, 1.0, 1.0};
  pred = {0.1, 0.4, 0.35, 0.8};
  metric.Accumulate(Y, pred);
  metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, 0.75);
  EXPECT_EQ(metric.metric_type(), "AUC");
}

TEST(MAEMetricTest, mae_test) {
  std::vector<real_t> Y;
  Y.push_back(12);
  Y.push_back(13);
  Y.push_back(14);
  Y.push_back(15);
  std::vector<real_t> pred;
  pred.push_back(11);
  pred.push_back(12);
  pred.push_back(13);
  pred.push_back(14);
  MAEMetric metric;
  size_t threadNumber = std::thread::hardware_concurrency();
  ThreadPool* pool = new ThreadPool(threadNumber);
  metric.Initialize(pool);
  metric.Accumulate(Y, pred);
  real_t metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, 1.0);
  metric.Reset();
  Y[0] = 23;
  Y[1] = 24;
  Y[2] = 25;
  Y[3] = 26;
  pred[0] = 12;
  pred[1] = 12;
  pred[2] = 12;
  pred[3] = 12;
  metric.Accumulate(Y, pred);
  metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, 12.5);
  EXPECT_EQ(metric.metric_type(), "MAE");
}

TEST(MAPEMetricTest, mape_test) {
  std::vector<real_t> Y;
  Y.push_back(12);
  Y.push_back(12);
  Y.push_back(12);
  Y.push_back(12);
  std::vector<real_t> pred;
  pred.push_back(11);
  pred.push_back(11);
  pred.push_back(11);
  pred.push_back(11);
  MAPEMetric metric;
  size_t threadNumber = std::thread::hardware_concurrency();
  ThreadPool* pool = new ThreadPool(threadNumber);
  metric.Initialize(pool);
  metric.Accumulate(Y, pred);
  real_t metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, 0.0833333333);
  metric.Reset();
  Y[0] = 23;
  Y[1] = 23;
  Y[2] = 23;
  Y[3] = 23;
  pred[0] = 12;
  pred[1] = 12;
  pred[2] = 12;
  pred[3] = 12;
  metric.Accumulate(Y, pred);
  metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, 0.478260869);
  EXPECT_EQ(metric.metric_type(), "MAPE");
}

TEST(RMSDMetricTest, rsmd_test) {
  std::vector<real_t> Y;
  Y.push_back(2);
  Y.push_back(2);
  Y.push_back(2);
  Y.push_back(2);
  std::vector<real_t> pred;
  pred.push_back(1);
  pred.push_back(1);
  pred.push_back(1);
  pred.push_back(1);
  RMSDMetric metric;
  size_t threadNumber = std::thread::hardware_concurrency();
  ThreadPool* pool = new ThreadPool(threadNumber);
  metric.Initialize(pool);
  metric.Accumulate(Y, pred);
  real_t metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, 1.0);
  metric.Reset();
  Y[0] = 23;
  Y[1] = 23;
  Y[2] = 23;
  Y[3] = 23;
  pred[0] = 12;
  pred[1] = 12;
  pred[2] = 12;
  pred[3] = 12;
  metric.Accumulate(Y, pred);
  metric_val = metric.GetMetric();
  EXPECT_FLOAT_EQ(metric_val, sqrt(121));
  EXPECT_EQ(metric.metric_type(), "RMSD");
}

Metric* CreateMetric(const char* format_name) {
  return CREATE_METRIC(format_name);
}

TEST(MetricTest, Create_Metric) {
  EXPECT_TRUE(CreateMetric("acc") != NULL);
  EXPECT_TRUE(CreateMetric("prec") != NULL);
  EXPECT_TRUE(CreateMetric("recall") != NULL);
  EXPECT_TRUE(CreateMetric("f1") != NULL);
  EXPECT_TRUE(CreateMetric("auc") != NULL);
  EXPECT_TRUE(CreateMetric("mae") != NULL);
  EXPECT_TRUE(CreateMetric("mape") != NULL);
  EXPECT_TRUE(CreateMetric("rmsd") != NULL);
  EXPECT_TRUE(CreateMetric("") == NULL);
  EXPECT_TRUE(CreateMetric("unknow_name") == NULL);
}

}  // namespace xLearn
