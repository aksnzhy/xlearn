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
This file tests thread_pool.h file.
*/

#include "gtest/gtest.h"

#include "src/base/thread_pool.h"

void func(int id) {
  printf("Hello %i\n", id);
}

TEST(ThreadPoolTest, Print_test) {
  ThreadPool pool(4);
  for (int i = 0; i < 3; ++i) {
    pool.enqueue(std::bind(func, 1));
    pool.enqueue(std::bind(func, 2));
    pool.enqueue(std::bind(func, 3));
    pool.enqueue(std::bind(func, 4));
    pool.Sync(4);
    printf("Hello master\n");
  }
  printf("final\n");
  usleep(100);
}

int a1 = 0;
int a2 = 0;
int a3 = 0;
int a4 = 0;
int a5 = 0;

void Sum(int *val) {
  for (int i = 0; i < 5; ++i) {
    (*val)++;
  }
}

TEST(ThreadPoolTest, Sum_test) {
  ThreadPool pool(5);
  for (int i = 0; i < 3; ++i) {
    pool.enqueue(std::bind(Sum, &a1));
    pool.enqueue(std::bind(Sum, &a2));
    pool.enqueue(std::bind(Sum, &a3));
    pool.enqueue(std::bind(Sum, &a4));
    pool.enqueue(std::bind(Sum, &a5));
    pool.Sync(5);
  }
  int sum = a1 + a2 + a3 + a4 + a5;
  EXPECT_EQ(sum, 75);
}
