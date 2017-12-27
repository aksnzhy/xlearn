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

This file is the implementation of the logging facilities.
*/

#include "src/base/logging.h"

#include <stdlib.h>
#include <time.h>
#include <assert.h>

//------------------------------------------------------------------------------
// Logger
//------------------------------------------------------------------------------

std::ofstream Logger::info_log_file_;
std::ofstream Logger::warn_log_file_;
std::ofstream Logger::erro_log_file_;

const char *cross_plat_ctime_r(time_t *cur_time, char *buffer)
{
#ifdef _WIN32
	size_t bufsize = 26; // The str length is always 26 after formatting
	errno_t e = ctime_s(buffer, bufsize, cur_time);
	//assert(e == 0 && "Huh? ctime_s returned an error");
	return buffer;
#else 
	const char *res = ctime_r(cur_time, buffer);
	//assert(res != NULL && "ctime_r failed...");
	return res;
#endif
}

void InitializeLogger(const std::string& info_log_filename,
                      const std::string& warn_log_filename,
                      const std::string& erro_log_filename) {
  Logger::info_log_file_.open(info_log_filename.c_str());
  Logger::warn_log_file_.open(warn_log_filename.c_str());
  Logger::erro_log_file_.open(erro_log_filename.c_str());
}

/*static*/
std::ostream& Logger::GetStream(LogSeverity severity) {
  if (severity == LogSeverity::INFO) {
    return info_log_file_.is_open() ? info_log_file_ : std::cout;
  } else if (severity == LogSeverity::WARNING) {
    return warn_log_file_.is_open() ? warn_log_file_ : std::cerr;
  } else if (severity == LogSeverity::ERR || severity == LogSeverity::FATAL) {
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
  cross_plat_ctime_r(&tm, time_string);
  return GetStream(severity) << time_string
                             << " " << file << ":" << line
                             << " (" << function << ") " << std::flush;
}

Logger::~Logger() {
  GetStream(severity_) << "\n" << std::flush;
  if (severity_ == LogSeverity::FATAL) {
    info_log_file_.close();
    warn_log_file_.close();
    erro_log_file_.close();
    abort();
  }
}
