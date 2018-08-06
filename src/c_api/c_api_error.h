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
This file defines the error handling for C API. 
*/

#ifndef C_API_C_API_ERROR_H_
#define C_API_C_API_ERROR_H_

#include <string>

#include "src/base/logging.h"
#include "src/c_api/c_api.h"

// Macro to guard begining and end section of all functions
#define API_BEGIN() try {

// Every function starts with API_BEGIN(); 
// and finishes with API_END() or API_END_HANDLE_ERROR
#define API_END()                                              \
  } catch(std::runtime_error &_except_) {                      \
    return XLAPIHandleException(_except_);                     \
} return 0;

// Every function starts with API_BEGIN(); 
// and finishes with API_END() or API_END_HANDLE_ERROR
// The finally clause contains procedure to cleanup states
// when an error happens.
#define API_END_HANDLE_ERROR(Finalize)                        \
  } catch(std::runtime_error &_except_) {                     \
    Finalize;                                                 \
    return XLAPIHandleException(_except_);                    \
} return 0;

// Get the last error message needed by C API
const char* XLearnGetLastError();

// Set the last error message needed by C API
void XLearnAPISetLastError(const char* msg);

// Handle exception thrown out and return value
// of API after exception is handled
inline int XLAPIHandleException(const std::runtime_error &e) {
  XLearnAPISetLastError(e.what());
  return -1;
}

#endif  // C_API_C_API_ERROR_H_
