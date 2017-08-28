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

#include <sys/utsname.h>
#include <unistd.h>

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
  //-------------------------------------------------------
  // Step 3: Init log file
  //-------------------------------------------------------
  std::string prefix = get_log_file();
  InitializeLogger(StringPrintf("%s.INFO", prefix.c_str()),
                   StringPrintf("%s.WARN", prefix.c_str()),
                   StringPrintf("%s.ERROR", prefix.c_str()));
  // For training
  if (hyper_param_.is_train) {
    //-------------------------------------------------------
    // Step 4: Init Reader and read problem
    //-------------------------------------------------------
    LOG(INFO) << "Start to init Reader";
    // Split file if use -cv
    if (hyper_param_.cross_validation) {
      CHECK_NE(hyper_param_.train_set_file.empty(), true);
      CHECK_GT(hyper_param_.num_folds, 0);
      splitor_.split(hyper_param_.train_set_file,
                     hyper_param_.num_folds);
      LOG(INFO) << "Split file into "
                << hyper_param_.num_folds
                << " parts.";
    }
    // Init Reader
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
    LOG(INFO) << "Number of Reader: " << num_reader;
    reader_.resize(num_reader, NULL);
    // Create Parser
    parser_ = create_parser();
    // Create Reader
    for (int i = 0; i < num_reader; ++i) {
      reader_[i] = create_reader();
      reader_[i]->Initialize(file_list[i],
                             hyper_param_.batch_size,
                             parser_);
      LOG(INFO) << "Init Reader: " << file_list[i];
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
      LOG(INFO) << "Number of feature: " << max_feat;
      LOG(INFO) << "Number of field: " << max_field;
      // return to the begining
      reader_[i]->Reset();
    }
    //-------------------------------------------------------
    // Step 5: Init model parameter
    //-------------------------------------------------------
    if (hyper_param_.score_func.compare("fm") == 0) {
      hyper_param_.num_param = max_feat + 1 +
                            max_feat * hyper_param_.num_K;
    } else if (hyper_param_.score_func.compare("ffm") == 0) {
      hyper_param_.num_param = max_feat + 1 +
                       max_feat * max_field * hyper_param_.num_K;
    } else { // linear socre
      hyper_param_.num_param = max_feat + 1;
    }
    LOG(INFO) << "Number parameters: " << hyper_param_.num_param;
    if (hyper_param_.score_func.compare("linear") == 0) {
      // Initialize all parameters to zero
      model_ = new Model(hyper_param_, false);
      LOG(INFO) << "Initialize model to zero.";
    } else {
      // Initialize parameters using Gaussian distribution
      model_ = new Model(hyper_param_, true);
      LOG(INFO) << "Initialize model using Gaussian distribution.";
    }
    //-------------------------------------------------------
    // Step 6: Init Updater
    //-------------------------------------------------------
    updater_ = create_updater();
    updater_->Initialize(hyper_param_);
    LOG(INFO) << "Initialize Updater.";
    //-------------------------------------------------------
    // Step 7: Init score function
    //-------------------------------------------------------
    score_ = create_score();
    score_->Initialize(hyper_param_);
    LOG(INFO) << "Initialize score function.";
    //-------------------------------------------------------
    // Step 8: Init loss function
    //-------------------------------------------------------
    loss_ = create_loss();
    loss_->Initialize(score_);
    LOG(INFO) << "Initialize loss function.";
  }
  // For inference
  if (!hyper_param_.is_train) {
    //-------------------------------------------------------
    // Step 4: Init Reader and reader problembal
    //-------------------------------------------------------
    // Create Parser
    parser_ = create_parser();
    // Create Reader
    reader_.resize(1, create_reader());
    CHECK_NE(hyper_param_.inference_file.empty(), true);
    reader_[0]->Initialize(hyper_param_.inference_file,
                           hyper_param_.batch_size,
                           parser_);
    LOG(INFO) << "Initialize Parser ans Reader.";
    //-------------------------------------------------------
    // Step 5: Init model parameter
    //-------------------------------------------------------
    model_ = new Model(hyper_param_.model_file);
    hyper_param_.score_func = model_->GetScoreFunction();
    hyper_param_.num_feature = model_->GetNumFeature();
    if (hyper_param_.score_func.compare("fm") == 0 ||
        hyper_param_.score_func.compare("ffm") == 0) {
      hyper_param_.num_K = model_->GetNumK();
    }
    if (hyper_param_.score_func.compare("ffm") == 0) {
      hyper_param_.num_field = model_->GetNumField();
    }
    LOG(INFO) << "Initialize model.";
    //-------------------------------------------------------
    // Step 6: Init score function
    //-------------------------------------------------------
    score_ = create_score();
    score_->Initialize(hyper_param_);
    LOG(INFO) << "Initialize score function.";
    //-------------------------------------------------------
    // Step 7: Init loss function
    //-------------------------------------------------------
    loss_ = create_loss();
    loss_->Initialize(score_);
    LOG(INFO) << "Initialize score function.";
  }
}

// Start training or inference
void Solver::StartWork() {
  if (hyper_param_.is_train) {
    LOG(INFO) << "Start training work.";
    start_train_work();
  } else {
    LOG(INFO) << "Start inference work.";
    start_inference_work();
  }
}

// Finalize xLearn
void Solver::Finalize() {
  if (hyper_param_.is_train) {
    finalize_train_work();
  } else {
    finalize_inference_work();
  }
}

// Train
void Solver::start_train_work() {

}

void Solver::finalize_train_work() {
  LOG(INFO) << "Finalize training work.";
}

// Inference
void Solver::start_inference_work() {

}

void Solver::finalize_inference_work() {
  LOG(INFO) << "Finalize inference work."
}

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
    LOG(ERROR) << "Cannot create reader: " << str;
  }
  return reader;
}

// Create Updater by a given string
Updater* Solver::create_updater() {
  Updater* updater;
  updater = CREATE_UPDATER(hyper_param_.updater_type.c_str());
  if (updater == NULL) {
    LOG(ERROR) << "Cannot create updater: "
               << hyper_param_.updater_type;
  }
  return updater;
}

// Create Score by a given string
Score* Solver::create_score() {
  Score* score;
  score = CREATE_SCORE(hyper_param_.score_func.c_str());
  if (score == NULL) {
    LOG(ERROR) << "Cannot create score: "
               << hyper_param_.score_func;
  }
  return score;
}

// Create Loss by a given string
Loss* Solver::create_loss() {
  Loss* loss;
  loss = CREATE_LOSS(hyper_param_.loss_func.c_str());
  if (loss == NULL) {
    LOG(ERROR) << "Cannot create loss: "
               << hyper_param_.loss_func;
  }
  return loss;
}

// Find max feature in a data matrix
index_t Solver::find_max_feature(DMatrix* matrix, int num_samples) {
  index_t res = 0;
  for (int i = 0; i < num_samples; ++i) {
    SparseRow* row = matrix->row[i];
    for (int j = 0; j < row->column_len; ++j) {
      if (row->idx[j] > res) {
        res = row->idx[j];
      }
    }
  }
  return res;
}

// Find max field in a data matrix
index_t Solver::find_max_field(DMatrix* matrix, int num_samples) {
  index_t res = 0;
  for (int i = 0; i < num_samples; ++i) {
    SparseRow* row = matrix->row[i];
    for (int j = 0; j < row->column_len; ++j) {
      if (row->field[j] > res) {
        res = row->field[j];
      }
    }
  }
  return res;
}

// Get host name
std::string Solver::get_host_name() {
  struct utsname buf;
  if (0 != uname(&buf)) {
    *buf.nodename = '\0';
  }
  return std::string(buf.nodename);
}

// Get user name
std::string Solver::get_user_name() {
  const char* username = getenv("USER");
  return username != NULL ? username : getenv("USERNAME");
}

// Get current system time
std::string Solver::print_current_time() {
  time_t current_time = time(NULL);
  struct tm broken_down_time;
  CHECK(localtime_r(&current_time, &broken_down_time) == &broken_down_time);
  return StringPrintf("%04d%02d%02d-%02d%02d%02d",
                      1900 + broken_down_time.tm_year,
                      1 + broken_down_time.tm_mon,
                      broken_down_time.tm_mday,
                      broken_down_time.tm_hour,
                      broken_down_time.tm_min,
                      broken_down_time.tm_sec);
}

// The log file name = base + host_name + username +
//                     date_time + process_id
std::string Solver::get_log_file() {
  CHECK(!hyper_param_.log_file.empty());
  std::string filename_prefix;
  SStringPrintf(&filename_prefix,
                "%s.%s.%s.%s.%u",
                hyper_param_.log_file.c_str(),
                get_host_name().c_str(),
                get_user_name().c_str(),
                print_current_time().c_str(),
                getpid());
  return filename_prefix;
}

} // namespace xLearn
