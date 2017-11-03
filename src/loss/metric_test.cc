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
}

TEST(RSMDMetricTest, rsmd_test) {
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
  EXPECT_FLOAT_EQ(metric_val, sqrt(0.006944444444));
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
  EXPECT_FLOAT_EQ(metric_val, sqrt(0.228733459357));
}

}  // namespace xLearn