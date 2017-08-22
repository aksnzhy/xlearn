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
#include <cstdio>

#include "src/base/stringprintf.h"

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
  printf(""
"----------------------------------------------------------------------------\n"
               "      _\n"
               "     | |\n"
               "__  _| |     ___  __ _ _ __ _ __\n"
               "\\ \\/ / |    / _ \\/ _` | '__| '_ \\ \n"
               " >  <| |___|  __/ (_| | |  | | | |\n"
               "/_/\\_\\_____/\\___|\\__,_|_|  |_| |_|\n\n"
               "   xLearn   -- 0.10 Version --\n"
"----------------------------------------------------------------------------\n"
  );
}

// Initialize Trainer
void Solver::Initialize(int argc, char* argv[]) {
  //-------------------------------------------------------
  // Step 1: Print logo
  //-------------------------------------------------------
  print_logo();
  //-------------------------------------------------------
  // Step 2: Check and parse command line arguments
  //-------------------------------------------------------
  checker_.Initialize(argc, argv);
  if (!checker_.Check(hyper_param_)) {
    printf("Arguments error \n");
    exit(0);
  }
  // For training
  if (hyper_param_.is_train) {
    //-------------------------------------------------------
    // Step 3: Init Reader and read problem
    //-------------------------------------------------------
    // Split file if use -cv
    if (hyper_param_.cross_validation) {
      CHECK_NE(hyper_param_.train_set_file.empty(), true);
      CHECK_GT(hyper_param_.num_folds, 0);
      splitor_.split(hyper_param_.train_set_file,
                     hyper_param_.num_folds);
    }
    // Init Reader
    // First get file list
    int num_reader = 0;
    std::vector<std::string> file_list;
    if (hyper_param_.cross_validation) {
      num_reader += hyper_param_.num_folds;
      for (int i = 0; i < hyper_param_.num_folds; ++i) {
        std::string filename = StringPrintf("%s_%d",
                                 hyper_param_.train_set_file.c_str(),
                                 i);
        file_list.push_back(filename);
      }
    } else { // do not use CV
      num_reader += 1;
      CHECK_NE(hyper_param_.train_set_file.empty(), true);
      file_list.push_back(hyper_param_.train_set_file);
      if (!hyper_param_.test_set_file.empty()) {
        num_reader += 1;
        file_list.push_back(hyper_param_.test_set_file);
      }
    }
    // Create Parser
    parser_ = create_parser();
    // Create Reader
    for (int i = 0; i < num_reader; ++i) {
      reader_[i] = create_reader();
      reader_[i]->Initialize(file_list[i],
                             hyper_param_.batch_size,
                             parser_);
    }
    // Read problem and init some hyper-parameters
    DMatrix* matrix = NULL;
    index_t max_feat = 0, max_field = 0;
    for (int i = 0; i < num_reader; ++i) {
      int num_samples = 0;
      do {
        num_samples = reader_[i]->Samples(matrix);
        int tmp = find_max_feature(matrix, num_samples);
        if (tmp > max_feat) {
          max_feat = tmp;
        }
        if (hyper_param_.score_func.compare("ffm") == 0) {
          tmp = find_max_field(matrix, num_samples);
          if (tmp > max_field) {
            max_field = tmp;
          }
        }
      } while (num_samples != 0);
      hyper_param_.num_feature = max_feat;
      hyper_param_.num_field = max_field;
      // return to the begining
      reader_[i]->Reset();
    }
    //-------------------------------------------------------
    // Step 4: Init model parameter
    //-------------------------------------------------------

    //-------------------------------------------------------
    // Step 4: Init Updater
    //-------------------------------------------------------
  }
  // For inference
  if (!hyper_param_.is_train) {

  }
}

// Start training or inference
void Solver::StartWork() { }

// Finalize xLearn
void Solver::Finalize() { }

// Create Parser by a given string
Parser* Solver::create_parser() {
  Parser* parser;
  parser = CREATE_PARSER(hyper_param_.file_format.c_str());
  if (parser == NULL) {
    LOG(ERROR) << "Cannot create parser: "
               << hyper_param_.file_format;
  }
  return parser;
}

// Create Reader by a given string
Reader* Solver::create_reader() {
  Reader* reader;
  std::string str = hyper_param_.on_disk ? "disk" : "memory";
  reader = CREATE_READER(str.c_str());
  if (reader == NULL) {
    LOG(ERROR) << "Cannot create reader: "
               << str;
  }
  return reader;
}

// Find max feature in a data matrix
index_t Solver::find_max_feature(DMatrix* matrix, int num_samples) {
  index_t res = 0;
  for (int i = 0; i < num_samples; ++i) {

  }
  return res;
}

// Find max field in a data matrix
index_t Solver::find_max_field(DMatrix* matrix, int num_samples) {
  index_t res = 0;
  for (int i = 0; i < num_samples; ++i) {

  }
  return res;
}

} // namespace xLearn
