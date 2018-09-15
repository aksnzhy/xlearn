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
This file defines facilities for format printing.
*/

#ifndef XLEARN_BASE_FORMAT_PRINT_H_
#define XLEARN_BASE_FORMAT_PRINT_H_

#include <iostream>
#include <vector>
#include <string>

#include "src/base/common.h"

namespace Color {

#define IMPORTANT_MSG true
#define NOT_IMPORTANT_MSG false

// Color table
enum Code {
/*********************************************************
 *  Color for front                                      *
 *********************************************************/
  FG_RED      = 31,
  FG_GREEN    = 32,
  FG_YELLOW   = 33,
  FG_BLUE     = 34,
  FG_MAGENTA  = 35,
  FG_CYAN     = 36,
  FG_WHITE    = 37,
  FG_DEFAULT  = 39,
/*********************************************************
 *  Color for Background                                 *
 *********************************************************/
  BG_RED      = 41,
  BG_GREEN    = 42,
  BG_YELLOW   = 43,
  BG_BLUE     = 44,
  BG_MAGENTA  = 45,
  BG_CYAN     = 46,
  BG_WHITE    = 47,
  BG_DEFAULT  = 49,
/*********************************************************
 *  Control code                                         *
 *********************************************************/  
  RESET          = 0,   // everything back to normal
  BOLD           = 1,   // often a brighter shade of the same colour
  UNDER_LINE     = 4,   
  INVERSE        = 7,   // swap forground and background color
  BOLD_OFF       = 21,
  UNDER_LINE_OFF = 24,
  INVERSE_OFF    = 27
};

//------------------------------------------------------------------------------
// The Modifier class can be used to set the foreground and background
// color of output. We can this class like this:
// 
//   Modifier red(Color::FG_RED);
//   Modifier def(Color::FG_DEFAULT);
//   cout << "This ->" << red << "word" << def "<- is red. " << endl;
//------------------------------------------------------------------------------
class Modifier {
  Code code;
 public:
  Modifier(Code pCode) : code(pCode) {}
  friend std::ostream&
  operator<<(std::ostream& os, const Modifier& mod) {
  	return os << "\033[" << mod.code << "m";
  }
 private:
  DISALLOW_COPY_AND_ASSIGN(Modifier);
};

// [Warning] blablabla ...
inline void print_warning(const std::string& out) {
  Color::Modifier mag(Color::FG_MAGENTA);
  Color::Modifier bold(Color::BOLD);
  Color::Modifier reset(Color::RESET);
  std::cout << mag << bold << "[ WARNING    ] " 
            << out << reset << std::endl;
}

inline void print_error(const std::string& out) {
  Color::Modifier red(Color::FG_RED);
  Color::Modifier bold(Color::BOLD);
  Color::Modifier reset(Color::RESET);
  std::cout << red << bold << "[ ERROR      ] "
            << out << reset << std::endl;
}

inline void print_action(const std::string& out) {
  Color::Modifier green(Color::FG_GREEN);
  Color::Modifier bold(Color::BOLD);
  Color::Modifier reset(Color::RESET);
  std::cout << green << bold << "[ ACTION     ] "
            << out << reset << std::endl;
}

inline void print_info(const std::string& out, bool important = false) {
  Color::Modifier green(Color::FG_GREEN);
  Color::Modifier bold(Color::BOLD);
  Color::Modifier reset(Color::RESET);
  if (!important) {
    std::cout << green << "[------------] " << reset
            << out << std::endl;
  } else {
    std::cout << green << bold << "[------------] " << out << reset
              << std::endl;
  }
}

//------------------------------------------------------------------------------
// Example:
//
//  column ->  "Name", "ID", "Count", "Price"
//  width -> 10, 10, 10, 10
//
// Output:
//
//   Name       ID        Count   Price
//   Fruit      0x101       50     5.27  
//   Juice      0x102       20     8.73  
//   Meat       0x104       30    10.13
//------------------------------------------------------------------------------
template <typename T>
void print_row(const std::vector<T>& column, 
               const std::vector<int>& width) {
  CHECK_EQ(column.size(), width.size());
  for (size_t i = 0; i < column.size(); ++i) {
    std::cout.width(width[i]);
    std::cout << column[i];
  }
  std::cout << "\n";
}

}  // namespace Color

#endif  // XLEARN_BASE_FORMAT_PRINT_H_