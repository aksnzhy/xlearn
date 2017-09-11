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

This file is the implementation of Parser.
*/

#include "src/reader/parser.h"

#include <stdlib.h>

#include "src/base/split_string.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_parser_registry, Parser);
REGISTER_PARSER("libsvm", LibsvmParser);
REGISTER_PARSER("libffm", FFMParser);
REGISTER_PARSER("csv", CSVParser);

//------------------------------------------------------------------------------
// LibsvmParser parses the following data format:
// [y1 idx:value idx:value ...]
// [y2 idx:value idx:value ...]
//------------------------------------------------------------------------------
void LibsvmParser::Parse(const StringList& list, DMatrix& matrix) {
  CHECK_GE(list.size(), 0);
  CHECK_GE(matrix.row_len, 0);
  size_t row_len = matrix.row_len;
  for (size_t i = 0; i < row_len; ++i) {
    int col_len = 1;
    // Add Y
    char *y_char = strtok(const_cast<char*>(list[i].data()), " \t");
    // Add bias
    matrix.Y[i] = atof(y_char);
    matrix.row[i]->idx.push_back(0);
    matrix.row[i]->X.push_back(1.0);
    // Add the other feature
    for (;;) {
      char *idx_char = strtok(nullptr, ":");
      char *value_char = strtok(nullptr, " \t");
      if(idx_char == NULL || *idx_char == '\n') {
        break;
      }
      matrix.row[i]->idx.push_back(atoi(idx_char));
      matrix.row[i]->X.push_back(atof(value_char));
      ++col_len;
    }
    matrix.row[i]->column_len = col_len;
  }
}

//------------------------------------------------------------------------------
// FFMParser parses the following data format:
// [y1 field:idx:value field:idx:value ...]
// [y2 field:idx:value field:idx:value ...]
//------------------------------------------------------------------------------
void FFMParser::Parse(const StringList& list, DMatrix& matrix) {
  CHECK_GE(list.size(), 0);
  CHECK_GE(matrix.row_len, 0);
  size_t row_len = matrix.row_len;
  for (size_t i = 0; i < row_len; ++i) {
    int col_len = 1;
    // Add Y
    char *y_char = strtok(const_cast<char*>(list[i].data()), " \t");
    // Add bias
    matrix.Y[i] = atof(y_char);
    matrix.row[i]->field.push_back(0);
    matrix.row[i]->idx.push_back(0);
    matrix.row[i]->X.push_back(1.0);
    // Add the other feature
    for (;;) {
      char *field_char = strtok(nullptr, ":");
      char *idx_char = strtok(nullptr, ":");
      char *value_char = strtok(nullptr, " \t");
      if(field_char == NULL || *field_char == '\n') {
        break;
      }
      matrix.row[i]->field.push_back(atoi(field_char));
      matrix.row[i]->idx.push_back(atoi(idx_char));
      matrix.row[i]->X.push_back(atof(value_char));
      ++col_len;
    }
    matrix.row[i]->column_len = col_len;
  }
}

//------------------------------------------------------------------------------
// CSVParser parses the following data format:
// [y1 value value value ...]
// [y2 value value value ...]
//------------------------------------------------------------------------------
void CSVParser::Parse(const StringList& list, DMatrix& matrix) {
  static StringList m_items;
  static StringList m_single_item;
  CHECK_GE(list.size(), 0);
  CHECK_GE(matrix.row_len, 0);
  size_t row_len = matrix.row_len;
  for (size_t i = 0; i < row_len; ++i) {
    m_items.clear();
    SplitStringUsing(list[i], " \t", &m_items);
    matrix.Y[i] = atof(m_items[0].c_str());
    // Get real length of current row
    int len = 0;
    for (int j = 1; j < m_items.size(); ++j) {
      if (atof(m_items[j].c_str()) != 0) {
        ++len;
      }
    }
    CHECK_NOTNULL(matrix.row[i]);
    matrix.row[i]->Resize(len+1); // add a bias
    matrix.row[i]->idx[0] = 0;
    matrix.row[i]->X[0] = 1.0;
    int k = 1;
    for (int j = 1; j < m_items.size(); ++j) {
      real_t value = atof(m_items[j].c_str());
      if (value != 0) {
        matrix.row[i]->idx[k] = j;
        matrix.row[i]->X[k] = value;
        ++k;
      }
    }
  }
}

} // namespace xLearn
