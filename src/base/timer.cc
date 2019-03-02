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
This file is the implementation of the Timer class.
*/

#include "src/base/timer.h"

Timer::Timer() {
  reset();
}

// Reset code start
void Timer::reset() {
  begin = std::chrono::high_resolution_clock::now();
  duration =
     std::chrono::duration_cast<std::chrono::milliseconds>(begin-begin);
}

// Code start
void Timer::tic() {
  begin = std::chrono::high_resolution_clock::now();
}

// Code end
float Timer::toc() {
  duration += std::chrono::duration_cast<std::chrono::milliseconds>
              (std::chrono::high_resolution_clock::now()-begin);
  return get();
}

// Get the time duration (seconds)
float Timer::get() {
  return (float)duration.count() / 1000;
}