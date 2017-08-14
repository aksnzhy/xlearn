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

This file defines the StrSimilar class, which can be used to
find similar strings.
*/

#ifndef XLEARN_BASE_LEVEN_SHTEIN_DISTANCE_H_
#define XLEARN_BASE_LEVEN_SHTEIN_DISTANCE_H_

#include <vector>
#include <string>

#include "src/base/common.h"

namespace xLearn {

typedef std::vector<std::string> StringList;

//------------------------------------------------------------------------------
// StrSimilar class is used to find the similar string from the
// target string list.
//------------------------------------------------------------------------------
class StrSimilar {
 public:
  // Constructor and Destructor
  StrSimilar() { }
  ~StrSimilar() { }

  // Return true (false) if str is (not) in target string list
  bool Find(const std::string& str,
            const StringList& list);

  // Find the most similar string from string list
  // Return the minimal levenshtein distance
  int FindSimilar(const std::string& str,
                  const StringList& list,
                  std::string& result);

 private:
  // Levenshtein distance
  int ldistance(const std::string& source,
                const std::string& target);

  DISALLOW_COPY_AND_ASSIGN(StrSimilar);
};

} // namespace xLearn

#endif // XLEARN_BASE_LEVEN_SHTEIN_DISTANCE_H_
