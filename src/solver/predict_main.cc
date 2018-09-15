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
This file is the entry for prediction of the xLearn.
*/

#include "src/base/common.h"
#include "src/base/timer.h"
#include "src/base/stringprintf.h"
#include "src/solver/solver.h"

//------------------------------------------------------------------------------
// The pre-defined main function
//------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  Timer timer;
  timer.tic();

  xLearn::Solver solver;
  solver.SetPredict();
  solver.Initialize(argc, argv);
  solver.StartWork();
  solver.Clear();

  Color::print_info(
    StringPrintf("Total time cost: %.2f (sec)", 
    timer.toc()), false);

  return 0;
}
