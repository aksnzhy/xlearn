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
This file is the implementation of Parser class.
*/

#include "src/reader/parser.h"

#include "src/base/split_string.h"

#include <stdlib.h>

namespace xLearn {

// Max size of one line TXT data
static const uint32 kMaxLineSize = 10 * 1024 * 1024;  // 10 MB

static char line_buf[kMaxLineSize];

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_parser_registry, Parser);
REGISTER_PARSER("libsvm", LibsvmParser);
REGISTER_PARSER("libffm", FFMParser);
REGISTER_PARSER("csv", CSVParser);

// Get one line from memory buffer
uint64 Parser::get_line_from_buffer(char* line,
                                    char* buf,
                                    uint64 pos,
                                    uint64 size) {
  if (pos >= size) { return 0; }
  uint64 end_pos = pos;
  while (end_pos < size && buf[end_pos] != '\n') { end_pos++; }
  uint64 read_size = end_pos - pos + 1;
  if (read_size > kMaxLineSize) {
    LOG(FATAL) << "Encountered a too-long line.    \
                   Please check the data.";
  }
  memcpy(line, buf+pos, read_size);
  line[read_size - 1] = '\0';
  if (read_size > 1 && line[read_size - 2] == '\r') {
    // Handle some txt format in windows or DOS.
    line[read_size - 2] = '\0';
  }
  return read_size;
}



//------------------------------------------------------------------------------
// LibsvmParser parses the following data format:
// [y1 idx:value idx:value ...]
// [y2 idx:value idx:value ...]
// idx can start from 0
//------------------------------------------------------------------------------
void LibsvmParser::Parse(char* buf, 
                         uint64 size, 
                         DMatrix& matrix, 
                         bool reset) {
  CHECK_NOTNULL(buf);
  CHECK_GT(size, 0);
  // Clear the data matrix
  if (reset) { 
    matrix.Reset(); 
  }
  // Parse every line
  uint64 pos = 0;
  for (;;) {
    uint64 rd_size = get_line_from_buffer(line_buf, buf, pos, size);
    if (rd_size == 0) break;
    pos += rd_size;
    matrix.AddRow();
    int i = matrix.row_length - 1;
    if (reset == false) {
    }
    // Add Y
    if (has_label_) {  // for training task
      char *y_char = strtok(line_buf, splitor_.c_str());
      matrix.Y[i] = atof(y_char);
    } else {  // for predict task
      matrix.Y[i] = -2;
    }
    // Add features
    real_t norm = 0.0;
    // The first element
    if (!has_label_) {
      char *idx_char = strtok(line_buf, ":");
      char *value_char = strtok(nullptr, splitor_.c_str());
      if (idx_char != nullptr && *idx_char != '\n') {
        index_t idx = atoi(idx_char);
        real_t value = atof(value_char);
        matrix.AddNode(i, idx, value);
        norm += value*value;
      }
    }
    // The remain elements
    for (;;) {
      char *idx_char = strtok(nullptr, ":");
      char *value_char = strtok(nullptr, splitor_.c_str());
      if (idx_char == nullptr || *idx_char == '\n') {
        break;
      }
      index_t idx = atoi(idx_char);
      real_t value = atof(value_char);
      matrix.AddNode(i, idx, value);
      norm += value*value;
    }
    norm = 1.0f / norm;
    matrix.norm[i] = norm;
  }
}

//------------------------------------------------------------------------------
// FFMParser parses the following data format:
// [y1 field:idx:value field:idx:value ...]
// [y2 field:idx:value field:idx:value ...]
// idx can start from 0
//------------------------------------------------------------------------------
void FFMParser::Parse(char* buf, 
                      uint64 size, 
                      DMatrix& matrix,
                      bool reset) {
  CHECK_NOTNULL(buf);
  CHECK_GT(size, 0);
  // Clear the data matrix
  if (reset) { 
    matrix.Reset(); 
  }
  // Parse every line
  uint64 pos = 0;
  for (;;) {
    uint64 rd_size = get_line_from_buffer(line_buf, buf, pos, size);
    if (rd_size == 0) break;
    pos += rd_size;
    matrix.AddRow();
    int i = matrix.row_length - 1;
    // Add Y
    if (has_label_) {  // for training task
      char *y_char = strtok(line_buf, splitor_.c_str());
      matrix.Y[i] = atof(y_char);
    } else {  // for predict task
      matrix.Y[i] = -2;
    }
    // Add features
    real_t norm = 0.0;
    // The first element
    if (!has_label_) {
      char *field_char = strtok(line_buf, ":");
      char *idx_char = strtok(nullptr, ":");
      char *value_char = strtok(nullptr, splitor_.c_str());
      if (idx_char != nullptr && *idx_char != '\n') {
        index_t idx = atoi(idx_char);
        real_t value = atof(value_char);
        index_t field_id = atoi(field_char);
        matrix.AddNode(i, idx, value, field_id);
        norm += value*value;
      }
    }
    // The remain elements
    for (;;) {
      char *field_char = strtok(nullptr, ":");
      char *idx_char = strtok(nullptr, ":");
      char *value_char = strtok(nullptr, splitor_.c_str());
      if (field_char == nullptr || *field_char == '\n') {
        break;
      }
      index_t idx = atoi(idx_char);
      real_t value = atof(value_char);
      index_t field_id = atoi(field_char);
      matrix.AddNode(i, idx, value, field_id);
      norm += value*value;
    }
    norm = 1.0f / norm;
    matrix.norm[i] = norm;
  }
}

//------------------------------------------------------------------------------
// CSVParser parses the following data format:
// [y1 feat_1 feat_2 feat_3 ... feat_n]
// [y2 feat_1 feat_2 feat_3 ... feat_n]
// Note that, if the csv file doesn't contain the
// label y, users should add a placeholder to the dataset
// by themselves (Also in test data). Otherwise, the parser 
// will treat the first element as the label y.
//------------------------------------------------------------------------------
void CSVParser::Parse(char* buf, 
                      uint64 size, 
                      DMatrix& matrix, 
                      bool reset) {
  CHECK_NOTNULL(buf);
  CHECK_GT(size, 0);
  // Clear the data matrix
  if (reset) { 
    matrix.Reset(); 
  }
  // Parse every line
  uint64 pos = 0;
  std::vector<std::string> str_vec;
  for (;;) {
    uint64 rd_size = get_line_from_buffer(line_buf, buf, pos, size);
    if (rd_size == 0) break;
    pos += rd_size;
    matrix.AddRow();
    int i = matrix.row_length - 1;
    str_vec.clear();
    SplitStringUsing(line_buf, splitor_.c_str(), &str_vec);
    int size = str_vec.size();
    // Add Y
    matrix.Y[i] = atof(str_vec[0].c_str());
    // Add features
    real_t norm = 0.0;
    for (int j = 1; j < size; ++j) {
      index_t idx = j-1;
      real_t value = atof(str_vec[j].c_str());
      matrix.AddNode(i, idx, value);
      norm += value*value;
    }
    norm = 1.0f / norm;
    matrix.norm[i] = norm;
  }
}

} // namespace xLearn
