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

This file defines the facilities for format printing.
*/

#ifndef XLEARN_BASE_FORMAT_PRINT_H_
#define XLEARN_BASE_FORMAT_PRINT_H_

#include <iostream>
#include <vector>
#include <string>

#include "src/base/common.h"

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

//------------------------------------------------------------------------------
// Example:
//
//  std -> "Hello World !"
//
// Output:
//
//  -----------------
//  | Hello World ! |
//  -----------------
//------------------------------------------------------------------------------
inline void print_block(const std::string& str) {
  CHECK_NE(str.empty(), true);
  // Add two space and two lines
  size_t size = str.size() + 4;
  for (size_t i = 0; i < size; ++i) {
  	std::cout << "-";
  }
  std::cout << "\n";
  std::cout << "| ";
  std::cout << str;
  std::cout << " |";
  for (size_t i = 0; i < size; ++i) {
  	std::cout << "-";
  }
  std::cout << "\n";
}

#endif  // XLEARN_BASE_FORMAT_PRINT_H_