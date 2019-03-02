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
This file provides logging facilities treating log 
messages by their severities.
*/

#ifndef XLEARN_BASE_LOGGING_H_
#define XLEARN_BASE_LOGGING_H_

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

//------------------------------------------------------------------------------
// If function |InitializeLogger| was invoked and was able to open
// files specified by the parameters, log messages of various severity
// will be written into corresponding files. Otherwise, all log messages
// will be written to stderr. For example:
//
//   int main() {
//     InitializeLogger("/tmp/info.log", "/tmp/warn.log", "/tmp/erro.log");
//     LOG(INFO)    << "An info message going into /tmp/info.log";
//     LOG(WARNING) << "An warn message going into /tmp/warn.log";
//     LOG(ERR)     << "An erro message going into /tmp/erro.log";
//     LOG(FATAL)   << "An fatal message going into /tmp/erro.log, "
//                  << "and kills current process by a segmentation fault.";
//     return 0;
//   }
//------------------------------------------------------------------------------

void InitializeLogger(const std::string& info_log_filename,
                      const std::string& warn_log_filename,
                      const std::string& erro_log_filename);

enum LogSeverity { INFO, WARNING, ERR, FATAL };

class Logger {
  friend void InitializeLogger(const std::string& info_log_filename,
                               const std::string& warn_log_filename,
                               const std::string& erro_log_filename);
 public:
  Logger(LogSeverity s) : severity_(s) {}
  ~Logger();

  static std::ostream& GetStream(LogSeverity severity);
  
  static std::ostream& Start(LogSeverity severity,
                             const std::string& file,
                             int line,
                             const std::string& function);

 private:
  static std::ofstream info_log_file_;
  static std::ofstream warn_log_file_;
  static std::ofstream erro_log_file_;
  LogSeverity severity_;
};

//-----------------------------------------------------------------------------
// The basic mechanism of logging.{h,cc} is as follows:
//  - LOG(severity) defines a Logger instance, which records the severity.
//  - LOG(severity) then invokes Logger::Start(), which invokes Logger::Stream
//    to choose an output stream, outputs a message head into the stream and
//    flush.
//  - The std::ostream reference returned by LoggerStart() is then passed to
//    user-specific output operators (<<), which writes the log message body.
//  - When the Logger instance is destructed, the destructor appends flush.
//    If severity is FATAL, the destructor causes SEGFAULT and core dump.
//
// It is important to flush in Logger::Start() after outputing message
// head.  This is because that the time when the destructor is invoked
// depends on how/where the caller code defines the Logger instance.
// If the caller code crashes before the Logger instance is properly
// destructed, the destructor might not have the chance to append its
// flush flags.  Without flush in Logger::Start(), this may cause the
// lose of the last few messages.  However, given flush in Start(),
// program crashing between invocations to Logger::Start() and
// destructor only causes the lose of the last message body; while the
// message head will be there.
//-----------------------------------------------------------------------------
#define LOG(severity)                                                       \
  Logger(severity).Start(severity, __FILE__, __LINE__, __FUNCTION__)

#endif   // XLEARN_BASE_LOGGING_H_
