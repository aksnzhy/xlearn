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
This file defines the Reader class that is responsible for
reading data from data source.
*/

#ifndef XLEARN_READER_READER_H_
#define XLEARN_READER_READER_H_

#include <string>
#include <vector>
#include <thread>
#include <algorithm>

#include "src/base/common.h"
#include "src/base/class_register.h"
#include "src/base/scoped_ptr.h"
#include "src/base/thread_pool.h"
#include "src/data/data_structure.h"
#include "src/reader/parser.h"

namespace xLearn {

const int kDefautBlockSize = 500;  // 500 MB

//------------------------------------------------------------------------------
// Reader is an abstract class which can be implemented in different way,
// such as the InmemReader that reads data from memory, and the OndiskReader
// that reads data from disk file for large-scale machine learning.
//
// We can use the Reader class like this (Pseudo code):
//
//   #include "reader.h"
//
//   /* or new InmemReader() */
//   Reader* reader = new OndiskReader();
//
//   /* For in-memory Reader, the buffer_size (MB) will not be used */
//   /* For on-disk Reader, the shuffle will always be false */
//  
//   reader->Initialize(filename = "/tmp/testdata",
//                      buffer_size = 200,
//                      shuffle = false);
//  
//   DMatrix* matrix = nullptr;
// 
//   Loop {
//
//      size_t num_samples = reader->Sample(DMatrix);
//
//      /* The reader will return 0 when reaching the end of
//      data source, and then we can invoke Reset() to return
//      to the begining of data */
//  
//      if (num_samples == 0) {
//        reader->Reset()
//      }
//
//      /* use data matrix ... */
//
//   }
//
// For now, the Reader can parse three kinds of file format, including
// the libsvm format, the libffm format, and the CSV format.
//------------------------------------------------------------------------------
class Reader {
 public:
  // Constructor and Desstructor
  Reader() : shuffle_(false) {  }
  virtual ~Reader() {  }

  // We need to invoke the Initialize() function before
  // we start to sample data. We can shuffle data before 
  // training, and this is good for SGD.
  virtual void Initialize(const std::string& filename) = 0;

  // Sample data from disk or from memory buffer.
  // Return the number of record in each samplling.
  // Samples() will return 0 when reaching end of the data.
  virtual index_t Samples(DMatrix* &matrix) = 0;

  // Return to the begining of the data.
  virtual void Reset() = 0;

  // Free the memory of data matrix.
  virtual void Clear() = 0;

  // Return the Reader type
  virtual std::string Type() = 0;

  // This method is only used in On-disk Reader
  virtual void SetBlockSize(int size) = 0;

  // Wether current dataset has label y ?
  bool inline has_label() { return has_label_; }

  // If shuffle data ?
  virtual void SetShuffle(bool shuffle) {
    this->shuffle_ = shuffle;
  }

 protected:
  /* Input file name */
  std::string filename_;
  /* Sample() returns this data sample */
  DMatrix data_samples_;
  /* Parser for a block of memory buffer */
  Parser* parser_;
  /* If this data has label y ?
  This value will be set automitically
  in initialization */
  bool has_label_;
  /* If shuffle data ? */
  bool shuffle_;

  // Check current file format and return
  // "libsvm", "ffm", or "csv".
  // Program crashes for unknow format.
  // This function will also check if current
  // data has the label y.
  std::string check_file_format();

  // Create parser for different file format
  Parser* CreateParser(const char* format_name) {
    return CREATE_PARSER(format_name);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(Reader);
};

//------------------------------------------------------------------------------
// Sampling data from memory buffer.
// For in-memory smaplling, the Reader will automatically convert
// txt data to binary data, and uses this binary data in the next time.
//------------------------------------------------------------------------------
class InmemReader : public Reader {
 public:
  // Constructor and Destructor
  InmemReader() : pos_(0) { }
  ~InmemReader() { }

  // Pre-load all the data into memory buffer.
  virtual void Initialize(const std::string& filename);

  // Sample data from the memory buffer.
  virtual index_t Samples(DMatrix* &matrix);

  // Return to the begining of the data.
  virtual void Reset();

  // Free the memory of data matrix.
  virtual void Clear() {
    data_buf_.Release();
  }

  // Return the Reader type
  virtual std::string Type() {
    return "in-memory";
  }

  // This method is only used in On-Disk Reader
  virtual void SetBlockSize(int size) {
    // Do nothing
    return;
  }

  // If shuffle data ?
  virtual inline void SetShuffle(bool shuffle) {
    this->shuffle_ = shuffle;
    if (shuffle_ && !order_.empty()) {
      random_shuffle(order_.begin(), order_.end());
    }
  }

  // Get data buffer
  virtual inline DMatrix* GetMatrix() {
    return &data_buf_;
  }

 protected:
  /* Reader will load all the data 
  into this buffer */
  DMatrix data_buf_;
  /* Number of record at each samplling */
  index_t num_samples_;
  /* Position for samplling */
  index_t pos_;
  /* For random shuffle */
  std::vector<index_t> order_;

  // Check wheter current path has a binary file.
  bool hash_binary(const std::string& filename);

  // Initialize Reader from existing binary file.
  void init_from_binary();

  // Initialize Reader from a new txt file.
  void init_from_txt();

 private:
  DISALLOW_COPY_AND_ASSIGN(InmemReader);
};

//------------------------------------------------------------------------------
// Samplling data from disk file.
// OndiskReader is used to train very big data, which cannot be
// loaded into main memory of current single machine.
//------------------------------------------------------------------------------
class OndiskReader : public Reader {
 public:
  // Constructor and Destructor
  OndiskReader() 
    : block_size_(500) {  }  // 500 MB by default
  ~OndiskReader() { 
    Clear();
    Close(file_ptr_); 
  }

  // Create parser and open file
  virtual void Initialize(const std::string& filename);

  // Sample data from disk file
  virtual index_t Samples(DMatrix* &matrix);

  // Return to the head of file
  virtual void Reset();

  // Free the memory of data matrix.
  virtual void Clear() {
    data_samples_.Release();
    if (block_ != nullptr) {
      delete [] block_;
    }
  }

  // Return the Reader type
  virtual std::string Type() {
    return "on-disk";
  }

  // This method is only used in On-Disk Reader
  virtual void SetBlockSize(int size) {
    CHECK_GT(size, 0);
    block_size_ = size;
  }

  // We cannot set shuffle for OndiskReader
  void inline SetShuffle(bool shuffle) {
    if (shuffle == true) {
      LOG(ERR) << "Cannot set shuffle for OndiskReader.";
    }
    this->shuffle_ = false;
  }

  // Set block size. Note that we need to invoke this 
  // function before invoking the Initialize() function.    
  void SetBlockSize(size_t size) {
    CHECK_GT(size, 0);
    this->block_size_ = size;
  }
 
 protected:
  /* Maintain the file pointer */
  FILE* file_ptr_;
  /* A block of memory to store the data */
  char* block_;
  /* Block size */
  size_t block_size_;

  // Find the last '\n' in block and 
  // shrink back file pointer.
  void shrink_block(char* block, size_t* ret, FILE* file);

 private:
  DISALLOW_COPY_AND_ASSIGN(OndiskReader);
};

// TODO(aksnzhy)
//------------------------------------------------------------------------------
// Copy DMatrix from some other data source, e.g., Python Pandas.
// When we use the Python interface of xLearn, users often need to 
// convert some other data object to a DMatrix data structure, and then
// we can use CopyReader to initialize the Reader class. After that, the
// CopyReader will act as the same with the InmemReader.
//------------------------------------------------------------------------------
class CopyReader : public Reader {
 public:
  // Constructor and Destructor
  CopyReader() : pos_(0) { }
  ~CopyReader() { }
  
  // We do nothing in this funtion
  virtual void Initialize(const std::string& filename);

  // Copy DMatrix from the other data source
  virtual void CopyDMatrix(DMatrix* matrix);

  // Sample data from the memory buffer.
  virtual index_t Samples(DMatrix* &matrix);

  // Return to the head of file
  virtual void Reset();

  // Free the memory of data matrix.
  virtual void Clear() {
    data_buf_.Release();
  }

  // Return Reader type
  virtual std::string Type() {
    return "copy";
  }

  // This method is only used in On-Disk Reader
  virtual void SetBlockSize(int size) {
    // Do nothing
    return;
  }

  // If shuffle data ?
  virtual inline void SetShuffle(bool shuffle) {
    this->shuffle_ = shuffle;
    if (shuffle_ && !order_.empty()) {
      random_shuffle(order_.begin(), order_.end());
    }
  }

  // Get data buffer
  virtual inline DMatrix* GetMatrix() {
    return &data_buf_;
  }

 protected:
  /* Reader will load all the data 
  into this buffer */
  DMatrix data_buf_;
  /* Number of record at each samplling */
  index_t num_samples_;
  /* Position for samplling */
  index_t pos_;
  /* For random shuffle */
  std::vector<index_t> order_;

 private:
  DISALLOW_COPY_AND_ASSIGN(CopyReader);
};

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_DEFINE_REGISTRY(xLearn_reader_registry, Reader);

#define REGISTER_READER(format_name, reader_name)           \
  CLASS_REGISTER_OBJECT_CREATOR(                            \
      xLearn_reader_registry,                               \
      Reader,                                               \
      format_name,                                          \
      reader_name)

#define CREATE_READER(format_name)                          \
  CLASS_REGISTER_CREATE_OBJECT(                             \
      xLearn_reader_registry,                               \
      format_name)

} // namespace xLearn

#endif // XLEARN_READER_READER_H_
