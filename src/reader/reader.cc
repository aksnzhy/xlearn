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

// Check current file format and
// return 'libsvm', 'libffm', or 'csv'
// This function will also check if current
// data has the label y
std::string Reader::check_file_format() {
  FILE* file = OpenFileOrDie(filename_.c_str(), "r");
  // get the first line of data
  std::string data_line;
  GetLine(file, data_line);
  Close(file);
  // Split the first line of data
  std::vector<std::string> str_list;
  SplitStringUsing(data_line, " \t", &str_list);
  // has y?
  size_t found = str_list[0].find(":");
  if (found != std::string::npos) {  // find ":", no label
    has_label_ = false;
  } else {
    has_label_ = true;
  }
  // check file format
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
  CHECK_GE(num_samples, 0);
  filename_ = filename;
  num_samples_ = num_samples;
  printf("First check if the text file (%s) has been already "
         "converted to binary format \n", filename.c_str());
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
   *  Step 1: Init data_buf_                               *
   *********************************************************/
  data_buf_.Deserialize(filename_);
  has_label_ = data_buf_.has_label;
  /*********************************************************
   *  Step 3: Init data_samples_                           *
   *********************************************************/
  // For in-memory Reader, the size of data sample is
  // equals to the size of data buffer
  num_samples_ = data_buf_.row_length;
  data_samples_.ResetMatrix(num_samples_);
  /*********************************************************
   *  Step 3: Init order_                                  *
   *********************************************************/
  order_.resize(num_samples_);
  for (int i = 0; i < order_.size(); ++i) {
    order_[i] = i;
  }
  random_shuffle(order_.begin(), order_.end());
}

// Pre-load all the data to memory buffer from txt file
void InmemReader::init_from_txt() {
  /*********************************************************
   *  Step 1: Init parser_                                 *
   *********************************************************/
  parser_ = CreateParser(check_file_format().c_str());
  if (has_label_) parser_->setLabel(true);
  else parser_->setLabel(false);
  /*********************************************************
   *  Step 2: Init data_buf_                               *
   *********************************************************/
  char* buffer = nullptr;
  uint64 file_size = ReadFileToMemory(filename_, &buffer);
  printf("%s", PrintSize(file_size).c_str());
  parser_->Parse(buffer, file_size, data_buf_);
  data_buf_.SetHash(HashFile(filename_, true),
                    HashFile(filename_, false));
  data_buf_.has_label = has_label_;
  /*********************************************************
   *  Step 3: Init data_samples_                           *
   *********************************************************/
  num_samples_ = data_buf_.row_length;
  data_samples_.ResetMatrix(num_samples_, has_label_);
  /*********************************************************
   *  Step 4: order_                                       *
   *********************************************************/
  order_.resize(num_samples_);
  for (int i = 0; i < order_.size(); ++i) {
    order_[i] = i;
  }
  random_shuffle(order_.begin(), order_.end());
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
int InmemReader::Samples(DMatrix* &matrix, bool shuffle) {
  int num_line = 0;
  for (int i = 0; i < num_samples_; ++i) {
    if (pos_ >= data_buf_.row_length) {
      // End of the data buffer
      if (i == 0 && shuffle) {
        random_shuffle(order_.begin(), order_.end());
        matrix = nullptr;
        return 0;
      }
      break;
    }
    // Copy data between different DMatrix.
    data_samples_.row[i] = data_buf_.row[order_[pos_]];
    data_samples_.Y[i] = data_buf_.Y[order_[pos_]];
    data_samples_.norm[i] = data_buf_.norm[order_[pos_]];
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
}

// Sample data from disk file.
int OndiskReader::Samples(DMatrix* &matrix, bool shuffle) {
  return 0;
}

// Return to the begining of the file.
void OndiskReader::Reset() {}

}  // namespace xLearn
