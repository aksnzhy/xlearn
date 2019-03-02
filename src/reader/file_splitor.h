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
This file defines the FileSpliter which splits raw data file into K
subfiles so that K-folds validation can be easily applied under
our framework.
*/

#ifndef XLEARN_READER_FILE_SPLITER_H_
#define XLEARN_READER_FILE_SPLITER_H_

#include <string>

#include "src/base/common.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Split file using mmap() on Unix-like systems.
//------------------------------------------------------------------------------
class FileSpliter {
 public:
  FileSpliter() {  }
  ~FileSpliter() {  }

  void split(const std::string& filename, int num_blocks);

 private:
  DISALLOW_COPY_AND_ASSIGN(FileSpliter);
};

}  // namespace xLearn

#endif  // XLEARN_READER_FILE_SPLIT_H_
