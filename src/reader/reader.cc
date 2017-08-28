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

This file is the implementation of Reader.
*/

#include "src/reader/reader.h"

#include <string.h>
#include <algorithm> // for random_shuffle

#include "src/base/file_util.h"

static const uint32 kMaxLineSize = 100 * 1024; // 100 KB for one line of data

namespace xLearn {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_reader_registry, Reader);
REGISTER_READER("memory", InmemReader);
REGISTER_READER("disk", OndiskReader);

//------------------------------------------------------------------------------
// Implementation of InmemReader
//------------------------------------------------------------------------------

InmemReader::~InmemReader() {
  if (file_ptr_ != nullptr) {
    Close(file_ptr_);
  }
  data_buf_.Release(); // Release the in-memory buffer
}

// Pre-load all the data into memory buffer.
bool InmemReader::Initialize(const std::string& filename,
                             int num_samples,
                             Parser* parser) {
  CHECK_NE(filename.empty(), true)
  CHECK_GT(num_samples, 0);
  CHECK_NOTNULL(parser);
  filename_ = filename;
  num_samples_ = num_samples;
  parser_ = parser;
  file_ptr_ = OpenFileOrDie(filename_.c_str(), "r");
  if (file_ptr_ == NULL) { return false; }
  data_samples_.Resize(num_samples);
  uint64 file_size = GetFileSize(file_ptr_);
  LOG(INFO) << "Data file size: " << file_size << " bytes.";
  scoped_array<char> buffer;
  try {
    buffer.reset(new char[file_size]);
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for Reader.";
  }
  // Read all the data from file
  uint64 read_size = fread(buffer.get(),
                           1,
                           file_size,
                           file_ptr_);
  CHECK_EQ(read_size, file_size);
  // Initialize the DMatrix buffer
  int num_lines = GetLineNumber(buffer.get(), read_size);
  bool has_field = parser_->Type() == "libffm" ? true : false;
  data_buf_.Resize(num_lines);
  data_buf_.InitSparseRow(has_field);
  // Initialize order
  order_.resize(num_lines);
  for (int i = 0; i < num_lines; ++i) {
    order_[i] = i;
  }
  // Parse each line of data from the buffer
  StringList list(num_lines);
  scoped_array<char> line(new char[kMaxLineSize]);
  uint64 start_pos = 0;
  for (int i = 0; i < num_lines; ++i) {
    uint32 line_size = ReadLineFromMemory(line.get(),
                                          buffer.get(),
                                          start_pos,
                                          file_size);
    CHECK_NE(line_size, 0);
    start_pos += line_size;
    line[line_size - 1] = '\0';
    if (line_size > 1 && line[line_size - 2] == '\r') {
      // Handle some txt format in windows or DOS.
      line[line_size - 2] = '\0';
    }
    list[i].assign(line.get());
  }
  // Parse StringList to DMatrix.
  parser_->Parse(list, data_buf_);
  return true;
}

// Read one line of data from the memory buffer
uint64 InmemReader::ReadLineFromMemory(char* line,
                                       char* buf,
                                       uint64 start_pos,
                                       uint64 total_len) {
  // End of file
  if (start_pos >= total_len) {
    return 0;
  }
  uint64 end_pos = start_pos;
  while (*(buf + end_pos) != '\n') { ++end_pos; }
  uint64 read_size = end_pos - start_pos + 1;
  if (read_size > kMaxLineSize) {
    LOG(FATAL) << "Encountered a too-long line.    \
                   Please check the data.";
  }
  memcpy(line, buf + start_pos, read_size);
  return read_size;
}

// How many lines are there in current memory buffer?
int InmemReader::GetLineNumber(const char* buf, uint64 buf_size) {
  int num = 0;
  for (uint64 i = 0; i < buf_size; ++i) {
    if (buf[i] == '\n') num++;
  }
  return num;
}

// Smaple data from memory buffer.
int InmemReader::Samples(DMatrix* &matrix) {
  int num_line = 0;
  for (int i = 0; i < num_samples_; ++i) {
    if (pos_ >= data_buf_.row_len) {
      // End of the data buffer
      if (i == 0) {
        random_shuffle(order_.begin(), order_.end());
        matrix = nullptr;
        return 0;
      }
      break;
    }
    // Copy data between different DMatrix.
    data_samples_.row[i] = data_buf_.row[order_[pos_]];
    data_samples_.Y[i] = data_buf_.Y[order_[pos_]];
    pos_++;
    num_line++;
  }
  // The last data block
  if (num_line != num_samples_) {
    data_samples_.Setlength(num_line);
  }
  matrix = &data_samples_;
  return num_line;
}

// Return to the begining of the data buffer.
void InmemReader::Reset() { pos_ = 0; }

//------------------------------------------------------------------------------
// Implementation of OndiskReader.
//------------------------------------------------------------------------------

OndiskReader::~OndiskReader() {
  if (file_ptr_ != nullptr) {
    Close(file_ptr_);
  }
  data_samples_.Release(); // Release the data sample buffer
}

bool OndiskReader::Initialize(const std::string& filename,
                              int num_samples,
                              Parser* parser) {
  CHECK_NE(filename.empty(), true);
  CHECK_GT(num_samples, 0);
  CHECK_NOTNULL(parser);
  filename_ = filename;
  num_samples_ = num_samples;
  parser_ = parser;
  file_ptr_ = OpenFileOrDie(filename_.c_str(), "r");
  if (file_ptr_ == NULL) { return false; }
  bool has_field = parser_->Type() == "libffm" ? true : false;
  data_samples_.Resize(num_samples);
  data_samples_.InitSparseRow(has_field);
  return false;
}

// Sample data from disk file.
int OndiskReader::Samples(DMatrix* &matrix) {
  static scoped_array<char> line(new char[kMaxLineSize]);
  static StringList list(num_samples_);
  int num_lines = 0;
  for (int i = 0; i < num_samples_; ++i) {
    if (fgets(line.get(), kMaxLineSize, file_ptr_) == nullptr) {
      // Either ferror or feof.
      if (i == 0) {
        matrix = nullptr;
        return 0;
      }
      break;
    }
    int read_len = strlen(line.get());
    if (line[read_len - 1] != '\n') {
      LOG(FATAL) << "Encountered a too-long line.   \
                     Please check the data.";
    } else {
      line[read_len - 1] = '\0';
      // Handle the txt format in DOS and windows.
      if (read_len > 1 && line[read_len - 2] == '\r') {
        line[read_len - 2] = '\0';
      }
    }
    list[i].assign(line.get());
    num_lines++;
  }
  // The last data block
  if (num_lines != num_samples_) {
    data_samples_.Setlength(num_lines);
  }
  parser_->Parse(list, data_samples_);
  matrix = &data_samples_;
  return num_lines;
}

// Return to the begining of the file.
void OndiskReader::Reset() { fseek(file_ptr_, 0, SEEK_SET); }

} // namespace xLearn
