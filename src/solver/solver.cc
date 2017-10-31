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
"----------------------------------------------------------------------------------------------\n"
                    "           _\n"
                    "          | |\n"
                    "     __  _| |     ___  __ _ _ __ _ __\n"
                    "     \\ \\/ / |    / _ \\/ _` | '__| '_ \\ \n"
                    "      >  <| |___|  __/ (_| | |  | | | |\n"
                    "     /_/\\_\\_____/\\___|\\__,_|_|  |_| |_|\n\n"
                    "        xLearn   -- 0.10 Version --\n"
"----------------------------------------------------------------------------------------------\n"
"\n"
  );
}

/******************************************************************************
 * Functions for xlearn initialization                                        *
 ******************************************************************************/

// Initialize Solver
void Solver::Initialize(int argc, char* argv[]) {
  //  Print logo
  print_logo();
  // Check and parse command line arguments
  checker(argc, argv);
  // Initialize log file
  init_log();
  // Init train or predict
  if (hyper_param_.is_train) {
    init_train();
  } else {
    init_predict();
  }
}

// Check and parse command line arguments
void Solver::checker(int argc, char* argv[]) {
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
}

// Initialize log file
void Solver::init_log() {
  std::string prefix = get_log_file();
  if (hyper_param_.is_train) {
    prefix += "_train";
  } else {
    prefix += "_predict";
  }
  InitializeLogger(StringPrintf("%s.INFO", prefix.c_str()),
                StringPrintf("%s.WARN", prefix.c_str()),
                StringPrintf("%s.ERROR", prefix.c_str()));
}

// Initialize training task
void Solver::init_train() {
  /*********************************************************
   *  Init Reader                                          *
   *********************************************************/
  Timer timer;
  timer.tic();
  printf("--------------------\n");
  printf("| Read problem ... |\n");
  printf("--------------------\n");
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
  /*********************************************************
   *  Read problem                                         *
   *********************************************************/
  DMatrix* matrix = NULL;
  index_t max_feat = 0, max_field = 0;
  for (int i = 0; i < num_reader; ++i) {
    int num_samples = 0;
    while(1) {
      num_samples = reader_[i]->Samples(matrix, false);
      if (num_samples == 0) { break; }
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
    }
    // return to the begining of target file
    reader_[i]->Reset();
  }
  hyper_param_.num_feature = max_feat + 1;
  LOG(INFO) << "Number of feature: " << hyper_param_.num_feature;
  printf("  Number of Feature: %d \n", hyper_param_.num_feature);
  if (hyper_param_.score_func.compare("ffm") == 0) {
    hyper_param_.num_field = max_field + 1;
    LOG(INFO) << "Number of field: " << hyper_param_.num_field;
    printf("  Number of Field: %d \n", hyper_param_.num_field);
  }
  real_t time_cost = timer.toc();
  printf("  Time cost for reading problem: %.2f (sec) \n",
         time_cost);
  /*********************************************************
   *  Init Model                                           *
   *********************************************************/
  timer.reset();
  timer.tic();
  printf("------------------------\n");
  printf("| Initialize model ... |\n");
  printf("------------------------\n");
  // Initialize parameters
  model_ = new Model();
  model_->Initialize(hyper_param_.score_func,
                   hyper_param_.loss_func,
                   hyper_param_.num_feature,
                   hyper_param_.num_field,
                   hyper_param_.num_K,
                   hyper_param_.model_scale);
  index_t num_param = model_->GetNumParameter();
  hyper_param_.num_param = num_param;
  LOG(INFO) << "Number parameters: " << num_param;
  printf("  Model size: %.2f MB\n",
           (double) num_param / (1024.0 * 1024.0));
  time_cost = timer.toc();
  printf("  Time cost for model initial: %.2f (sec) \n",
         time_cost);
  /*********************************************************
   *  Init score function                                  *
   *********************************************************/
  score_ = create_score();
  score_->Initialize(hyper_param_.learning_rate,
                     hyper_param_.regu_lambda);
  LOG(INFO) << "Initialize score function.";
  /*********************************************************
   *  Init loss function                                   *
   *********************************************************/
  loss_ = create_loss();
  loss_->Initialize(score_, hyper_param_.norm);
  LOG(INFO) << "Initialize loss function.";
  /*********************************************************
   *  Init metric                                          *
   *********************************************************/
  metric_ = create_metric();
  metric_->Initialize(hyper_param_.metric);
  LOG(INFO) << "Initialize evaluation metric.";
}

// Initialize predict task
void Solver::init_predict() {
  /*********************************************************
   *  Read problem from model file                         *
   *********************************************************/
   printf("Load model from %s ...\n",
          hyper_param_.model_file.c_str());
   Timer timer;
   timer.tic();
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
   printf("  Loss function: %s \n", hyper_param_.loss_func.c_str());
   printf("  Score function: %s \n", hyper_param_.score_func.c_str());
   printf("  Number of Feature: %d \n", hyper_param_.num_feature);
   if (hyper_param_.score_func.compare("fm") == 0 ||
       hyper_param_.score_func.compare("ffm") == 0) {
     printf("  Number of K: %d\n", hyper_param_.num_K);
     if (hyper_param_.score_func.compare("ffm") == 0) {
       printf("  Number of field: %d\n", hyper_param_.num_field);
     }
   }
   real_t time_cost = timer.toc();
   printf("  Time cost for loading model: %.2f (sec) \n",
          time_cost);
   LOG(INFO) << "Initialize model.";
   /*********************************************************
    *  Init Reader and read problem                         *
    *********************************************************/
   printf("Read problem ... \n");
   timer.reset();
   timer.tic();
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
   time_cost = timer.toc();
   printf("  Time cost for reading problem: %.2f (sec) \n",
          time_cost);
   LOG(INFO) << "Initialize Reader: " << hyper_param_.predict_file;
   /*********************************************************
    *  Init score function                                  *
    *********************************************************/
   score_ = create_score();
   LOG(INFO) << "Initialize score function.";
   /*********************************************************
    *  Init loss function                                   *
    *********************************************************/
   loss_ = create_loss();
   loss_->Initialize(score_, hyper_param_.norm);
   LOG(INFO) << "Initialize score function.";
}

/******************************************************************************
 * Functions for xlearn start work                                            *
 ******************************************************************************/

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

// Train
void Solver::start_train_work() {
  int epoch = hyper_param_.num_epoch;
  bool early_stop = hyper_param_.early_stop;
  bool quiet = hyper_param_.quiet;
  bool save_model = hyper_param_.model_file == "none" ? false: true;
  Trainer trainer;
  trainer.Initialize(reader_,  /* Reader list */
                     epoch,
                     model_,
                     loss_,
                     metric_,
                     early_stop,
                     quiet);
  printf("----------------------\n");
  printf("| Start to train ... |\n");
  printf("----------------------\n");
  if (hyper_param_.cross_validation) {
    trainer.CVTrain();
    printf("-------------------\n");
    printf("| Finish training |\n");
    printf("-------------------\n");
  } else {
    trainer.Train();
    if (save_model) {
      Timer timer;
      timer.tic();
      printf("-----------------------------------------------\n");
      printf("| Finish training and start to save model ... |\n");
      printf("-----------------------------------------------\n");
      trainer.SaveModel(hyper_param_.model_file);
      real_t time_cost = timer.toc();
      printf("  Model file: %s\n", hyper_param_.model_file.c_str());
      printf("  Time cost for saving model: %.2f (sec) \n",
             time_cost);
    } else {
      printf("-------------------\n");
      printf("| Finish training |\n");
      printf("-------------------\n");
    }
  }
}

// Inference
void Solver::start_inference_work() {
  printf("Start to predict ...\n");
  Predictor pdc;
  pdc.Initialize(reader_[0],
            model_,
            loss_,
            hyper_param_.output_file);
  // Predict and write output
  pdc.Predict();
}

/******************************************************************************
 * Functions for xlearn finalization                                          *
 ******************************************************************************/

// Finalize xLearn
void Solver::FinalizeWork() {
  if (hyper_param_.is_train) {
    finalize_train_work();
  } else {
    finalize_inference_work();
  }
}

void Solver::finalize_train_work() {
  LOG(INFO) << "Finalize training work.";
}

void Solver::finalize_inference_work() {
  LOG(INFO) << "Finalize inference work.";
}

/******************************************************************************
 * The other helper functions                                                 *
 ******************************************************************************/

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
