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
#include "src/base/split_string.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_reader_registry, Reader);
REGISTER_READER("memory", InmemReader);
REGISTER_READER("disk", OndiskReader);

// Check current file format
// Return 'libsvm', 'libffm', or 'csv'
std::string Reader::check_file_format() {
  FILE* file = OpenFileOrDie(filename_.c_str(), "r");
  // get the first line of data
  std::string data_line;
  GetLine(file, data_line);
  Close(file);
  std::vector<std::string> str_list;
  SplitStringUsing(data_line, " \t", &str_list);
  int count = 0;
  for (int i = 0; i < str_list[1].size(); ++i) {
    if (str_list[1][i] == ':') {
      count++;
    }
  }
  if (count == 1) {
    return "libsvm";
  } else if (count == 2) {
    return "libffm";
  } else if (count == 0){
    return "csv";
  }
  printf("[Error] Unknow file format \n");
  exit(0);
}

//------------------------------------------------------------------------------
// Implementation of InmemReader
//------------------------------------------------------------------------------

// Pre-load all the data into memory buffer (data_buf)
// Note that this funtion will first check whether we can use
// the binary file. If not, reader will generate one automatically
void InmemReader::Initialize(const std::string& filename,
                             int num_samples) {
  CHECK_NE(filename.empty(), true)
  CHECK_GT(num_samples, 0);
  filename_ = filename;
  num_samples_ = num_samples;
  printf("First check if the text file has been already "
         "converted to binary format \n");
  // HashBinary() will read the first two hash value
  // and then check it whether equal to the hash value generated
  // by HashFile() function from current txt file
  if (hash_binary(filename_)) {
    printf("Binary file found. Skip converting text to binary \n");
    filename_ += ".bin";
    init_from_binary();
  } else {
    printf("Binary file NOT found. Convert text "
           "file to binary file \n");
    init_from_txt();
  }
}

// Check wheter current path has a binary file
// We use double check here. We first check a the hash value
// of a small data block, then check the all file.
bool InmemReader::hash_binary(const std::string& filename) {
  std::string bin_file = filename + ".bin";
  // If the ".bin" file does not exists, return false
  if (!FileExist(bin_file.c_str())) { return false; }
  FILE* file = OpenFileOrDie(bin_file.c_str(), "r");
  // Check the first hash value
  uint64 hash_1 = 0;
  ReadDataFromDisk(file, (char*)&hash_1, sizeof(hash_1));
  if (hash_1 != HashFile(filename, true)) {
    Close(file);
    return false;
  }
  // Check the second hash value
  uint64 hash_2 = 0;
  ReadDataFromDisk(file, (char*)&hash_2, sizeof(hash_2));
  if (hash_2 != HashFile(filename, false)) {
    Close(file);
    return false;
  }
  Close(file);
  return true;
}

// In-memory Reader can be initialized from binary file
void InmemReader::init_from_binary() {
  /*********************************************************
   *  Step 1: Init data_samples_                           *
   *********************************************************/
  data_samples_.ResetMatrix(num_samples_);
  /*********************************************************
   *  Step 2: Init data_buf_                               *
   *********************************************************/
  data_buf_.Deserialize(filename_);
  /*********************************************************
   *  Step 3: Init order_                                  *
   *********************************************************/
  order_.resize(data_buf_.row_length);
  for (int i = 0; i < order_.size(); ++i) {
    order_[i] = i;
  }
}

// Pre-load all the data to memory buffer from txt file
void InmemReader::init_from_txt() {
  /*********************************************************
   *  Step 1: Init data_samples_                           *
   *********************************************************/
  data_samples_.ResetMatrix(num_samples_);
  /*********************************************************
   *  Step 2: Init parser_                                 *
   *********************************************************/
  parser_ = CreateParser(check_file_format().c_str());
  /*********************************************************
   *  Step 3: Init data_buf_                               *
   *********************************************************/
  char* buffer = nullptr;
  uint64 file_size = ReadFileToMemory(filename_, &buffer);
  printf("%s", PrintSize(file_size).c_str());
  parser_->Parse(buffer, file_size, data_buf_);
  data_buf_.SetHash(HashFile(filename_, true),
                    HashFile(filename_, false));
  /*********************************************************
   *  Step 4: order_                                       *
   *********************************************************/
  order_.resize(data_buf_.row_length);
  for (int i = 0; i < order_.size(); ++i) {
    order_[i] = i;
  }
  /*********************************************************
   *  Step 5: Deserialize in-memory buffer to disk file    *
   *********************************************************/
  std::string bin_file = filename_ + ".bin";
  this->serialize_buffer(bin_file);
  /*********************************************************
   *  Step 6: Finalize                                     *
   *********************************************************/
  delete [] buffer;
}

// Smaple data from memory buffer.
int InmemReader::Samples(DMatrix* &matrix) {
  int num_line = 0;
  for (int i = 0; i < num_samples_; ++i) {
    if (pos_ >= data_buf_.row_length) {
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
  data_samples_.row_length = num_line;
  matrix = &data_samples_;
  return num_line;
}

// Return to the begining of the data buffer.
void InmemReader::Reset() { pos_ = 0; }

// Serialize DMatrix to a binary file
void InmemReader::serialize_buffer(const std::string& filename) {
  data_buf_.Serialize(filename);
}

//------------------------------------------------------------------------------
// Implementation of OndiskReader.
//------------------------------------------------------------------------------

void OndiskReader::Initialize(const std::string& filename,
                              int num_samples) {
  /*
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
  return false;*/
}

// Sample data from disk file.
int OndiskReader::Samples(DMatrix* &matrix) {
  /*
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
  return num_lines;*/
  return 0;
}

// Return to the begining of the file.
void OndiskReader::Reset() {
  /*
  fseek(file_ptr_, 0, SEEK_SET);
  data_samples_.Setlength(num_samples_);*/
}

}  // namespace xLearn
