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
This file is the implementation of StringSplit utilities.
*/

#include "src/base/split_string.h"

#include "src/base/common.h"

// In most cases, delim contains only one character.  In this case, we
// use CalculateReserveForVector to count the number of elements should
// be reserved in result vector, and thus optimize SplitStringUsing.
static int CalculateReserveForVector(const std::string& full, const char* delim) {
  int count = 0;
  if (delim[0] != '\0' && delim[1] == '\0') {
    // Optimize the common case where delim is a single character.
    char c = delim[0];
    const char* p = full.data();
    const char* end = p + full.size();
    while (p != end) {
      if (*p == c) {  // This could be optimized with hasless(v,1) trick.
        ++p;
      } else {
        while (++p != end && *p != c) {
          // Skip to the next occurence of the delimiter.
        }
        ++count;
      }
    }
  }
  return count;
}

void SplitStringUsing(const std::string& full,
                      const char* delim,
                      std::vector<std::string>* result) {
  CHECK(delim != NULL);
  CHECK(result != NULL);
  result->reserve(CalculateReserveForVector(full, delim));
  back_insert_iterator< std::vector<std::string> > it(*result);
  SplitStringToIteratorUsing(full, delim, it);
}

void SplitStringToSetUsing(const std::string& full,
                           const char* delim,
                           std::set<std::string>* result) {
  CHECK(delim != NULL);
  CHECK(result != NULL);
  simple_insert_iterator<std::set<std::string> > it(result);
  SplitStringToIteratorUsing(full, delim, it);
}
