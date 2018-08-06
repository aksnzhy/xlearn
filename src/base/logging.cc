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
This file is the implementation of logging facilities.
*/

#include "src/base/logging.h"

#include <stdlib.h>
#include <time.h>

//------------------------------------------------------------------------------
// Logger
//------------------------------------------------------------------------------

std::ofstream Logger::info_log_file_;
std::ofstream Logger::warn_log_file_;
std::ofstream Logger::erro_log_file_;

void InitializeLogger(const std::string& info_log_filename,
                      const std::string& warn_log_filename,
                      const std::string& erro_log_filename) {
  Logger::info_log_file_.open(info_log_filename.c_str());
  Logger::warn_log_file_.open(warn_log_filename.c_str());
  Logger::erro_log_file_.open(erro_log_filename.c_str());
  // check if the file has been open
  bool bo = false;
  if (Logger::info_log_file_.is_open() == 0) {
    std::cout << "Cannot create file: " << info_log_filename << ". "
              << "Please check that wether you need to "
              << "create a new directory. \n";
    bo = true;
  }
  if (Logger::warn_log_file_.is_open() == 0) {
    std::cout << "Cannot create file: " << warn_log_filename << ". "
              << "Please check that wether you need to "
              << "create a new directory. \n";
    bo = true;
  }
  if (Logger::erro_log_file_.is_open() == 0) {
    std::cout << "Cannot create file:  " << erro_log_filename << ". "
              << "Please check that wether you need to "
              << "create a new directory. \n";
    bo = true;
  }
  if (bo) { exit(0); }
}

/*static*/
std::ostream& Logger::GetStream(LogSeverity severity) {
  if (severity == INFO) {
    return info_log_file_.is_open() ? info_log_file_ : std::cout;
  } else if (severity == WARNING) {
    return warn_log_file_.is_open() ? warn_log_file_ : std::cerr;
  } else if (severity == ERR || severity == FATAL) {
    return erro_log_file_.is_open() ? erro_log_file_ : std::cerr;
  }
  return std::cout; // Print message
}

/*static*/
std::ostream& Logger::Start(LogSeverity severity,
                            const std::string& file,
                            int line,
                            const std::string& function) {
  time_t tm;
  time(&tm);
  char time_string[128];
  ctime_r(&tm, time_string);
  return GetStream(severity) << time_string
                             << " " << file << ":" << line
                             << " (" << function << ") " << std::flush;
}

Logger::~Logger() {
  GetStream(severity_) << "\n" << std::flush;
  if (severity_ == FATAL) {
    info_log_file_.close();
    warn_log_file_.close();
    erro_log_file_.close();
    abort();
  }
}
