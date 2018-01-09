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

This file defines several system functions.
*/

#ifndef XLEARN_BASE_SYSTEM_H_
#define XLEARN_BASE_SYSTEM_H_

#ifdef WIN32
#include "src/base/uname.h"
#else
#include <sys/utsname.h>
#endif

#include <string>

#include "src/base/unistd.h"
#include "src/base/common.h"
#include "src/base/stringprintf.h"

struct tm *localtime_safe(const time_t *timep, struct tm *result)
{
#ifdef WIN32
	localtime_s(result, timep);
	return result;
#else
  return localtime_r(timep, result);
#endif
}

// Get host name
std::string get_host_name() {
  struct utsname buf;
  if (0 != uname(&buf)) {
    *buf.nodename = '\0';
  }
  return std::string(buf.nodename);
}

// Get user name
std::string get_user_name() {
  const char* username = getenv("USER");
  return username != NULL ? username : getenv("USERNAME");
}

// Get current system time
std::string print_current_time() {
  time_t current_time = time(NULL);
  struct tm broken_down_time;
  CHECK(localtime_safe(&current_time, &broken_down_time) == &broken_down_time);
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
std::string get_log_file(const std::string& file_base) {
  CHECK(!file_base.empty());
  std::string filename_prefix;
  SStringPrintf(&filename_prefix,
                "%s.%s.%s.%s.%u",
                file_base.c_str(),
                get_host_name().c_str(),
                get_user_name().c_str(),
                print_current_time().c_str(),
                getpid());
  return filename_prefix;
}


#endif  // XLEARN_BASE_SYSTEM_H_