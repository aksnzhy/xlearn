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
  checker_.Initialize(hyper_param_.is_train, argc, argv);
  if (!checker_.Check(hyper_param_)) {
    printf("Arguments error \n");
    exit(0);
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
  printf("Read and parse data ... \n");
  LOG(INFO) << "Start to init Reader";
  // Split file if use -cv
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
  // Get file format and create Parser
  hyper_param_.file_format =
       get_file_format(hyper_param_.train_set_file);
  if ((hyper_param_.file_format == "libsvm" ||
       hyper_param_.file_format == "csv") &&
      hyper_param_.score_func == "ffm") {
    printf("[Error] Please use libffm file format for FFM task \n");
    exit(0);
  }
  parser_ = create_parser();
  parser_->SetSplitor(splitor_ch_);
  // Create Reader
  for (int i = 0; i < num_reader; ++i) {
    reader_[i] = create_reader();
    reader_[i]->Initialize(file_list[i],
                           hyper_param_.sample_size,
                           parser_);
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
    // return to the begining
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
  /*********************************************************
   *  Step 2: Init Model                                   *
   *********************************************************/
   printf("Initialize model ...\n");
   if (hyper_param_.score_func.compare("fm") == 0) {
     hyper_param_.num_param = hyper_param_.num_feature +
                              hyper_param_.num_feature *
                              hyper_param_.num_K;
   } else if (hyper_param_.score_func.compare("ffm") == 0) {
     hyper_param_.num_param = hyper_param_.num_feature +
                              hyper_param_.num_feature *
                              hyper_param_.num_field *
                              hyper_param_.num_K;
   } else { // linear socre
     hyper_param_.num_param = hyper_param_.num_feature;
   }
   LOG(INFO) << "Number parameters: " << hyper_param_.num_param;
   printf("  Model size: %.2f MB\n",
           (double) hyper_param_.num_param /
           (1024.0 * 1024.0));
   if (hyper_param_.score_func.compare("linear") == 0) {
     // Initialize all parameters to zero
     model_ = new Model();
     model_->Initialize(hyper_param_.num_param,
                   hyper_param_.score_func,
                   hyper_param_.loss_func,
                   hyper_param_.num_feature,
                   hyper_param_.num_field,
                   hyper_param_.num_K,
                   false);
     LOG(INFO) << "Initialize model to zero.";
   } else {
     // Initialize parameters using Gaussian distribution
     model_ = new Model();
     model_->Initialize(hyper_param_.num_param,
                   hyper_param_.score_func,
                   hyper_param_.loss_func,
                   hyper_param_.num_feature,
                   hyper_param_.num_field,
                   hyper_param_.num_K,
                   true);
     LOG(INFO) << "Initialize model using Gaussian distribution.";
   }
   /*********************************************************
    *  Step 3: Init Updater method                          *
    *********************************************************/
    updater_ = create_updater();
    updater_->Initialize(hyper_param_.learning_rate,
                     hyper_param_.regu_lambda,
                     hyper_param_.decay_rate_1,
                     hyper_param_.decay_rate_2,
                     hyper_param_.num_param);
    LOG(INFO) << "Initialize Updater.";
    /*********************************************************
     *  Step 4: Init score function                          *
     *********************************************************/
     score_ = create_score();
     score_->Initialize(hyper_param_.num_feature,
                     hyper_param_.num_K,
                     hyper_param_.num_field);
     LOG(INFO) << "Initialize score function.";
     /*********************************************************
      *  Step 5: Init loss function                           *
      *********************************************************/
      loss_ = create_loss();
      loss_->Initialize(score_);
      LOG(INFO) << "Initialize loss function.";
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
    // Get file format and create Parser
    hyper_param_.file_format =
         get_file_format(hyper_param_.inference_file);
    if (hyper_param_.file_format == "libsvm" &&
        hyper_param_.score_func == "ffm") {
        printf("[Error] Cannot give a libsvm file to FFM task \n");
        exit(0);
    }
    parser_ = create_parser();
    parser_->SetSplitor(splitor_ch_);
    // Create Reader
    reader_.resize(1, create_reader());
    CHECK_NE(hyper_param_.inference_file.empty(), true);
    reader_[0]->Initialize(hyper_param_.inference_file,
                           hyper_param_.sample_size,
                           parser_);
    if (reader_[0] == NULL) {
      printf("Cannot open the file %s\n",
             hyper_param_.inference_file.c_str());
      exit(0);
    }
    LOG(INFO) << "Initialize Parser ans Reader.";
    /*********************************************************
     *  Step 2: Init score function                          *
     *********************************************************/
     score_ = create_score();
     score_->Initialize(hyper_param_.num_feature,
                     hyper_param_.num_K,
                     hyper_param_.num_field);
     LOG(INFO) << "Initialize score function.";
     /*********************************************************
      *  Step 2: Init loss function                          *
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
    trainer.Initialize(reader_, /* reader list*/
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

// Get the file format
std::string Solver::get_file_format(const std::string& filename) {
  FILE* file = OpenFileOrDie(StringPrintf("%s",
                        filename.c_str()).c_str(), "r");
  // get the first line of data
  std::string data_line;
  GetLine(file, data_line);
  // check the splitor
  if (data_line.find(" ")) {
    splitor_ch_ = " ";
  } else if (data_line.find("\t")) {
    splitor_ch_ = "\t";
  } else {
    printf("[Error] The instance in %s must be divided by space or tab \n",
          filename.c_str());
    exit(0);
  }
  // file format
  std::vector<std::string> str_list;
  SplitStringUsing(data_line, splitor_ch_.c_str(), &str_list);
  int count = 0;
  for (int i = 0; i < str_list[1].size(); ++i) {
    if (str_list[1][i] == ':') {
      count++;
    }
  }
  if (count == 0) {
    return "csv";
  } else if (count == 1) {
    return "libsvm";
  } else if (count == 2) {
    return "libffm";
  } else {
    printf("[Error] Unknow file format \n");
    exit(0);
  }
  Close(file);
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
