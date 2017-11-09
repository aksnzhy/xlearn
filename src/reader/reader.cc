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
// return 'libsvm', 'libffm', or 'csv'.
// This function will also check if current
// data has the label y.
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

// Pre-load all the data into memory buffer (data_buf_).
// Note that this funtion will first check whether we 
// can use the existing binary file. If not, reader will 
// generate one automatically.
void InmemReader::Initialize(const std::string& filename) {
  CHECK_NE(filename.empty(), true)
  filename_ = filename;
  printf("First check if the text file (%s) has been already "
         "converted to binary format \n", filename.c_str());
  // HashBinary() will read the first two hash value
  // and then check it whether equal to the hash value generated
  // by HashFile() function from current txt file.
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

// Check wheter current path has a binary file.
// We use double check here, that is, we first check 
// the hash value of a small data block, then check the whole file.
bool InmemReader::hash_binary(const std::string& filename) {
  std::string bin_file = filename + ".bin";
  // If the ".bin" file does not exists, return false.
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

// In-memory Reader can be initialized from binary file.
void InmemReader::init_from_binary() {
  // Init data_buf_                               
  data_buf_.Deserialize(filename_);
  has_label_ = data_buf_.has_label;
  // Init data_samples_
  num_samples_ = data_buf_.row_length;
  data_samples_.ResetMatrix(num_samples_);
  // for shuffle
  order_.resize(num_samples_);
  for (int i = 0; i < order_.size(); ++i) {
    order_[i] = i;
  }
}

// Pre-load all the data to memory buffer from txt file.
void InmemReader::init_from_txt() {
  // Init parser_                                 
  parser_ = CreateParser(check_file_format().c_str());
  if (has_label_) parser_->setLabel(true);
  else parser_->setLabel(false);
  // Init data_buf_
  char* buffer = nullptr;
  uint64 file_size = ReadFileToMemory(filename_, &buffer);
  parser_->Parse(buffer, file_size, data_buf_);
  data_buf_.SetHash(HashFile(filename_, true),
                    HashFile(filename_, false));
  data_buf_.has_label = has_label_;
  // Init data_samples_ 
  num_samples_ = data_buf_.row_length;
  data_samples_.ResetMatrix(num_samples_, has_label_);
  // for shuffle
  order_.resize(num_samples_);
  for (int i = 0; i < order_.size(); ++i) {
    order_[i] = i;
  }
  // Deserialize in-memory buffer to disk file.
  std::string bin_file = filename_ + ".bin";
  data_buf_.Serialize(bin_file);
  delete [] buffer;
}

// Smaple data from memory buffer.
index_t InmemReader::Samples(DMatrix* &matrix) {
  for (int i = 0; i < num_samples_; ++i) {
    if (pos_ >= data_buf_.row_length) {
      // End of the data buffer
      if (i == 0) {
        if (shuffle_) {
          random_shuffle(order_.begin(), order_.end());
        }
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
  }
  matrix = &data_samples_;
  return num_samples_;
}

// Return to the begining of the data buffer.
void InmemReader::Reset() { pos_ = 0; }

//------------------------------------------------------------------------------
// Implementation of OndiskReader.
//------------------------------------------------------------------------------

// Find the last '\n', and shrink back file pointer
void shrink_block(char* block, size_t* ret, FILE* file_ptr) {
  // Find the last '\n'
  size_t index = *ret-1;
  while (block[index] != '\n') { index--; }
  // Shrink back file pointer
  fseek(file_ptr, index - *ret, SEEK_CUR);
  *ret = index + 1;
}

// Read a block of data from disk file
void read_block(OndiskReader* reader) {
  // Convert MB to byte
  uint64 read_byte = reader->get_block_size() * 1024 * 1024;
  // Continuously read data in loop
  for (;;) {
    std::unique_lock<std::mutex> lock(reader->mutex_);
    // Wait until the consumer finish its job
    while (!reader->full_) {
      reader->cond_not_full_.wait(lock);
    }
    // We can set the length of data_sample_ to -1
    // so that the reader will know that we have already
    // finish our job, and then the reader can break the loop.
    if (reader->get_sample()->row_length == -1) {
      return;
    }
    size_t ret = ReadDataFromDisk(reader->get_file_ptr(), 
                                  reader->get_block(), 
                                  read_byte);
    // At the end of file
    if (ret == 0) {
      // Return to the head of file.
      reader->Reset();
      // Set the length of data_sample_ to zero
      // so that we can know that we reach the end 
      // of current file.
      reader->get_sample()->row_length = 0;
    } else {
      // Find the last '\n', and shrink back file pointer
      shrink_block(reader->get_block(), &ret, reader->file_ptr_);
      // Parse block to data_sample_
      reader->parser_->Parse(reader->get_block(), 
         ret, 
         *reader->get_sample());
    }
    // notice the consumer thread
    reader->full_ = true;
    reader->cond_not_empty_.notify_one();
  }
}

void OndiskReader::Initialize(const std::string& filename) { 
  CHECK_NE(filename.empty(), true);
  this->filename_ = filename;
  // Init parser_                                 
  parser_ = CreateParser(check_file_format().c_str());
  if (has_label_) parser_->setLabel(true);
  else parser_->setLabel(false);
  // Allocate memory for block
  try {
    this->block_ = (char*)malloc(block_size_*1024*1024);
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for data  \
                   block. Block size: " 
               << block_size_ << "MB. "
               << "You set change the block size via configuration.";
  }
  file_ptr_ = OpenFileOrDie(filename_.c_str(), "r");
  // Pick up one thread from thread pool as back-end thread
  CHECK_NOTNULL(pool_);
  pool_->enqueue(std::bind(read_block, this));
}

// Return to the begining of the file
void OndiskReader::Reset() {
  int ret = fseek(file_ptr_, 0, SEEK_SET);
  if (ret != 0) {
    LOG(FATAL) << "Fail to return to the head of file.";
  }
}

// Sample data from disk file.
index_t OndiskReader::Samples(DMatrix* &matrix) {
  std::unique_lock<std::mutex> lock(mutex_);
  // Wait until the producer finish its job
  while (this->full_) {
    cond_not_empty_.wait(lock);
  }
  if (data_samples_.row_length == 0) {
    cond_not_full_.notify_one();
    return 0; 
  } else {
    matrix->CopyFrom(&data_samples_);
    cond_not_full_.notify_one();
    return data_samples_.row_length;
  }
}

}  // namespace xLearn
