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

This file defines the Reader class that is responsible for
reading data from data source.
*/

#ifndef XLEARN_READER_READER_H_
#define XLEARN_READER_READER_H_

#include <string>
#include <vector>

#include "src/base/common.h"
#include "src/base/class_register.h"
#include "src/base/scoped_ptr.h"
#include "src/data/data_structure.h"
#include "src/reader/parser.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Reader is an abstract class which can be implemented in different way,
// such as the InmemReader that reads data from memory, and the OndiskReader
// that reads data from disk file (for limited memory).
//
// We can use the Reader class like this (Pseudo code):
//
//   #include "reader.h"
//
//   /* or new InmemReader() */
//   Reader* reader = new OndiskReader();
//
//   reader->Initialize(filename = "/tmp/testdata",
//                      num_samples = 20000);  // size of the working set
//
//   Loop {
//
//      int num_samples = reader->Sample(DMatrix);
//
//      /* The reader will return 0 when reaching the end of
//      data source, and then we can invoke Reset() to return
//      to the begining of data */
//      if (num_samples == 0) {
//        reader->Reset()
//      }
//
//      /* use data matrix ... */
//
//   }
//
// For now, the Reader can parse three kinds of file format, including
// libsvm format, libffm format, and CSV format.
//------------------------------------------------------------------------------
class Reader {
 public:
  // Constructor and Desstructor
  Reader() {  }
  virtual ~Reader() {  }

  // We need to invoke the Initialize() function before
  // we start to sample data
  virtual void Initialize(const std::string& filename,
                          int num_samples) = 0;

  // Sample data from disk or from memory buffer
  // Return the number of record in each samplling
  // Samples() return 0 when reaching end of the data
  // We can shuffle data randomly, this is good for SGD
  virtual int Samples(DMatrix* &matrix, bool shuffle = true) = 0;

  // Return to the begining of the data
  virtual void Reset() = 0;

  // If current dataset has label y
  bool inline has_label() { return has_label_; }

 protected:
  /* Input file name */
  std::string filename_;
  /* Number of data samples in working set */
  int num_samples_;
  /* Data sample */
  DMatrix data_samples_;
  /* Parser for a block of memory buffer */
  Parser* parser_;
  /* If this data has label y?
  This value will be set automitically
  in initialization */
  bool has_label_;

  // Check current file format and return
  // "libsvm", "ffm", or "csv".
  // Program crashes for unknow format
  // This function will also check if current
  // data has the label y
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
// txt data to binary data, and use this binary data in the next time.
// Reader will randomly shuffle the data during samplling.
//------------------------------------------------------------------------------
class InmemReader : public Reader {
 public:
  InmemReader() { pos_ = 0; }
  ~InmemReader() { }

  // Pre-load all the data into memory buffer
  // The num_samples will be setted to the line size
  // of current data automatically
  virtual void Initialize(const std::string& filename,
                          int num_samples = 0);

  // Sample data from the memory buffer
  virtual int Samples(DMatrix* &matrix, bool shuffle = true);

  // Return to the begining of the data
  virtual void Reset();

 protected:
  /* Reader will load all the data
  into this buffer */
  DMatrix data_buf_;
  /* Position for samplling */
  index_t pos_;
  /* For random shuffle */
  std::vector<index_t> order_;

  // Check wheter current path has a binary file
  bool hash_binary(const std::string& filename);

  // Initialize Reader from existing binary file
  void init_from_binary();

  // Initialize Reader from a new txt file
  void init_from_txt();

  // Serialize in-memory buffer to disk file
  void serialize_buffer(const std::string& filename);

 private:
  DISALLOW_COPY_AND_ASSIGN(InmemReader);
};

//------------------------------------------------------------------------------
// Samplling data from disk file.
// OndiskReader is used to train very big data, which cannot be
// loaded into main memory of current single machine.
// We use multi-thread to support data pipeline reading for
// better performance
//------------------------------------------------------------------------------
class OndiskReader : public Reader {
 public:
  OndiskReader() {  }
  ~OndiskReader() { }

  virtual void Initialize(const std::string& filename,
                          int num_samples);

  // Sample data from disk file
  virtual int Samples(DMatrix* &matrix, bool shuffle = true);

  // Return to the begining of the file
  virtual void Reset();

 private:
  DISALLOW_COPY_AND_ASSIGN(OndiskReader);
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
