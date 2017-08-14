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
This file is the implementation of the Trainer class.
*/

#include "src/solver/solver.h"

#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

namespace xLearn {

//------------------------------------------------------------------------------
//         _
//        | |
//   __  _| |     ___  __ _ _ __ _ __
//   \ \/ / |    / _ \/ _` | '__| '_ \
//    >  <| |___|  __/ (_| | |  | | | |
//   /_/\_\______\___|\__,_|_|  |_| |_|
//
//      xLearn   -- 0.10 Version --
//------------------------------------------------------------------------------
void Solver::print_logo() const {
  std::cout <<
"----------------------------------------------------------------------------\n"
            << "      _\n"
            << "     | |\n"
            << "__  _| |     ___  __ _ _ __ _ __\n"
            << "\\ \\/ / |    / _ \\/ _` | '__| '_ \\ \n"
            << " >  <| |___|  __/ (_| | |  | | | |\n"
            << "/_/\\_\\_____/\\___|\\__,_|_|  |_| |_|\n\n"
            << "   xLearn   -- 0.10 Version --\n"
            <<
"----------------------------------------------------------------------------\n";
}

// Initialize Trainer
void Solver::Initialize(int argc, char* argv[]) {
  // Print logo
  print_logo();
  // Check and parse command line arguments
  checker_.Initialize(argc, argv);
  if (!checker_.Check(hyper_param_)) {
    exit(0);
  }
}

// Start training or inference
void Solver::StartWork() { }

// Finalize xLearn
void Solver::Finalize() { }

} // namespace xLearn
