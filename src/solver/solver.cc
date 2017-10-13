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
This file is the implementation of the Solver class.
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
#include "src/base/split_string.h"

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
"\n"
  );
}

// Initialize Solver
void Solver::Initialize(int argc, char* argv[]) {
  //  Print logo
  print_logo();
  // Check and parse command line arguments
  try {
    checker_.Initialize(hyper_param_.is_train, argc, argv);
    if (!checker_.Check(hyper_param_)) {
      printf("Arguments error \n");
      exit(0);
    }
  } catch (std::invalid_argument &e) {
    printf("%s\n", e.what());
    exit(1);
  }
  // Initialize log file
  std::string prefix = get_log_file();
  if (hyper_param_.is_train) {
    prefix += "_train";
  } else {
    prefix += "_predict";
  }
  InitializeLogger(StringPrintf("%s.INFO", prefix.c_str()),
                StringPrintf("%s.WARN", prefix.c_str()),
                StringPrintf("%s.ERROR", prefix.c_str()));
  // Init train or predict
  if (hyper_param_.is_train) {
    init_train();
  } else {
    init_predict();
  }
}

// Initialize training task
void Solver::init_train() {
  /*********************************************************
   *  Step 1: Init Reader and read problem                 *
   *********************************************************/
  clock_t start, end;
  start = clock();
  printf("Read problem ... \n");
  LOG(INFO) << "Start to init Reader";
  // Split file if use -c
  if (hyper_param_.cross_validation) {
    CHECK_GT(hyper_param_.num_folds, 0);
    splitor_.split(hyper_param_.train_set_file,
                   hyper_param_.num_folds);
    LOG(INFO) << "Split file into "
              << hyper_param_.num_folds
              << " parts.";
  }
  // Get number of Reader and path of Reader
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
  } else { // do not use cross-validation
    num_reader++;
    CHECK_NE(hyper_param_.train_set_file.empty(), true);
    file_list.push_back(hyper_param_.train_set_file);
    if (!hyper_param_.test_set_file.empty()) {
      num_reader++;
      file_list.push_back(hyper_param_.test_set_file);
    }
  }
  LOG(INFO) << "Number of Reader: " << num_reader;
  reader_.resize(num_reader, NULL);
  // Create Reader
  for (int i = 0; i < num_reader; ++i) {
    reader_[i] = create_reader();
    reader_[i]->Initialize(file_list[i],
                           hyper_param_.sample_size);
    if (reader_[i] == NULL) {
      printf("Cannot open the file %s\n",
             file_list[i].c_str());
      exit(0);
    }
    LOG(INFO) << "Init Reader: " << file_list[i];
  }
  // Read problem and init some hyper_param
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
    // return to the begining of target file
    reader_[i]->Reset();
  }
  hyper_param_.num_feature = max_feat + 1; // add bias
  LOG(INFO) << "Number of feature: " << hyper_param_.num_feature;
  printf("  Number of Feature: %d \n", hyper_param_.num_feature);
  if (hyper_param_.score_func.compare("ffm") == 0) {
    hyper_param_.num_field = max_field + 1;
    LOG(INFO) << "Number of field: " << hyper_param_.num_field;
    printf("  Number of Field: %d \n", hyper_param_.num_field);
  }
  end = clock();
  printf("  Time cost for reading problem: %.2f sec \n",
    (float)(end-start) / CLOCKS_PER_SEC);
  /*********************************************************
   *  Step 2: Init Model                                   *
   *********************************************************/
   start = clock();
   printf("Initialize model ...\n");
   // Initialize parameters
   model_ = new Model();
   model_->Initialize(hyper_param_.score_func,
                   hyper_param_.loss_func,
                   hyper_param_.num_feature,
                   hyper_param_.num_field,
                   hyper_param_.num_K);
   index_t num_param = model_->GetNumParameter_w();
   LOG(INFO) << "Number parameters: " << num_param;
   printf("  Model size: %.2f MB\n",
           (double) num_param / (1024.0 * 1024.0));
   end = clock();
   printf("  Time cost for model initial: %.2f sec \n",
     (float)(end-start) / CLOCKS_PER_SEC);
   /*********************************************************
    *  Step 3: Init Updater method                          *
    *********************************************************/
    updater_ = create_updater();
    updater_->Initialize(hyper_param_.learning_rate,
                     hyper_param_.regu_lambda,
                     hyper_param_.decay_rate,
                     hyper_param_.num_param);
    LOG(INFO) << "Initialize Updater.";
    /*********************************************************
     *  Step 4: Init score function                          *
     *********************************************************/
    score_ = create_score();
    LOG(INFO) << "Initialize score function.";
    /*********************************************************
     *  Step 5: Init loss function                           *
     *********************************************************/
    loss_ = create_loss();
    loss_->Initialize(score_);
    LOG(INFO) << "Initialize loss function.";
    /*********************************************************
     *  Step 6: Init metric                                  *
     *********************************************************/
    metric_ = create_metric();
    metric_->Initialize(hyper_param_.metric);
    LOG(INFO) << "Initialize evaluation metric.";
}

// Initialize predict task
void Solver::init_predict() {
  /*********************************************************
   *  Step 1: Read problem from model file                 *
   *********************************************************/
   model_ = new Model(hyper_param_.model_file);
   hyper_param_.score_func = model_->GetScoreFunction();
   hyper_param_.loss_func = model_->GetLossFunction();
   hyper_param_.num_feature = model_->GetNumFeature();
   if (hyper_param_.score_func.compare("fm") == 0 ||
       hyper_param_.score_func.compare("ffm") == 0) {
     hyper_param_.num_K = model_->GetNumK();
   }
   if (hyper_param_.score_func.compare("ffm") == 0) {
     hyper_param_.num_field = model_->GetNumField();
   }
   LOG(INFO) << "Initialize model.";
   /*********************************************************
    *  Step 2: Init Reader and read problem                 *
    *********************************************************/
    // Create Reader
    reader_.resize(1, create_reader());
    CHECK_NE(hyper_param_.predict_file.empty(), true);
    reader_[0]->Initialize(hyper_param_.predict_file,
                           hyper_param_.sample_size);
    if (reader_[0] == NULL) {
      printf("Cannot open the file %s\n",
             hyper_param_.predict_file.c_str());
      exit(0);
    }
    LOG(INFO) << "Initialize Parser ans Reader.";
    /*********************************************************
     *  Step 3: Init score function                          *
     *********************************************************/
     score_ = create_score();
     LOG(INFO) << "Initialize score function.";
     /*********************************************************
      *  Step 4: Init loss function                           *
      *********************************************************/
      loss_ = create_loss();
      loss_->Initialize(score_);
      LOG(INFO) << "Initialize score function.";
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
void Solver::FinalizeWork() {
  if (hyper_param_.is_train) {
    finalize_train_work();
  } else {
    finalize_inference_work();
  }
}

// Train
void Solver::start_train_work() {
  int epoch = hyper_param_.num_epoch;
  bool early_stop = hyper_param_.early_stop;
  if (hyper_param_.cross_validation) {
    Trainer trainer;
    trainer.Initialize(reader_, /* reader list */
                       epoch,
                       model_,
                       loss_,
                       updater_,
                       early_stop);
    printf("Start to train ... \n");
    trainer.CVTrain();
    printf("Finish training. \n");
  } else { // do not use cv
    Trainer trainer;
    Reader* train_reader = reader_[0];
    Reader* test_reader = NULL;
    if (!hyper_param_.test_set_file.empty()) {
      test_reader = reader_[1];
    }
    trainer.Initialize(train_reader,
                       test_reader,
                       epoch,
                       model_,
                       loss_,
                       metric_,
                       updater_,
                       early_stop);
    printf("Start to train ... \n");
    trainer.Train();
    printf("Finish training and start to save model ...\n"
           "  Filename: %s\n",
           hyper_param_.model_file.c_str());
    trainer.SaveModel(hyper_param_.model_file);
  }
}

void Solver::finalize_train_work() {
  LOG(INFO) << "Finalize training work.";
}

// Inference
void Solver::start_inference_work() {

}

void Solver::finalize_inference_work() {
  LOG(INFO) << "Finalize inference work.";
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

// Create Metric
Metric* Solver::create_metric() {
  Metric* metric;
  metric = new Metric;
  return metric;
}

// Find max feature in a data matrix
index_t Solver::find_max_feature(DMatrix* matrix, int num_samples) {
  index_t res = 0;
  for (int i = 0; i < num_samples; ++i) {
    SparseRow* row = matrix->row[i];
    for (SparseRow::const_iterator iter = row->begin();
         iter != row->end(); ++iter) {
      if (iter->feat_id > res) {
        res = iter->feat_id;
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
    for (SparseRow::const_iterator iter = row->begin();
         iter != row->end(); ++iter) {
      if (iter->field_id > res) {
        res = iter->field_id;
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
