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
// that reads data from disk file.
//
// We can use the Reader class like this (Pseudo code):
//
//   #include "reader.h"
//
//   Reader* reader = new OndiskReader();
//
//   reader->Initialize(filename = "/tmp/testdata",  // the data path
//                      num_samples = 200,           // return N samples
//                      parser = libsvm_parser)      // data format
//
//   Loop {
//
//      int num_samples = reader->Sample(DMatrix);
//
//      // use data samples to train model ...
//
//   }
//
// The reader will return 0 when reaching the end of data source, and then
// we can invoke Reset() to return to the begining of data.
//------------------------------------------------------------------------------
class Reader {
 public:
  Reader() {  }
  virtual ~Reader() {  }

  // We need to invoke this method before we sample data.
  virtual bool Initialize(const std::string& filename,
                          int num_samples,
                          Parser* parser) = 0;

  // Sample data from disk or from memory buffer.
  // Return the number of record in each samplling.
  // Samples() return 0 when reaching end of the data.
  virtual int Samples(DMatrix* &matrix) = 0;

  // Return to the begining of the data.
  virtual void Reset() = 0;

 protected:
  std::string filename_;    // Indicate the input file
  int num_samples_;         // Number of data samples in each samplling
  FILE* file_ptr_;          // Maintain current file pointer
  DMatrix data_samples_;    // Data sample
  Parser* parser_;          // Parse StringList to DMatrix

 private:
  DISALLOW_COPY_AND_ASSIGN(Reader);
};

//------------------------------------------------------------------------------
// Sampling data from memory buffer.
// At each begining of sample we random shuffle the data.
//------------------------------------------------------------------------------
class InmemReader : public Reader {
 public:
  InmemReader() { pos_ = 0; }
  ~InmemReader();

  // Pre-load all the data into memory buffer.
  virtual bool Initialize(const std::string& filename,
                          int num_samples,
                          Parser* parser);

  // Sample data from memory buffer.
  virtual int Samples(DMatrix* &matrix);

  // Return to the begining of the data.
  virtual void Reset();

 protected:
  DMatrix data_buf_;             // Data buffer
  int pos_;                      // Position for samplling
  std::vector<index_t> order_;   // For shufflling

 private:
  // Counting the '\n' character
  int GetLineNumber(const char* buf, uint64 buf_size);

  // Read every line from memory buffer
  uint64 ReadLineFromMemory(char* line,
                            char* buf,
                            uint64 start_pos,
                            uint64 total_len);

  DISALLOW_COPY_AND_ASSIGN(InmemReader);
};

//------------------------------------------------------------------------------
// Samplling data from disk file.
//------------------------------------------------------------------------------
class OndiskReader : public Reader {
 public:
  OndiskReader() {  }
  ~OndiskReader();

  virtual bool Initialize(const std::string& filename,
                          int num_samples,
                          Parser* parser);

  // Sample data from disk file.
  virtual int Samples(DMatrix* &matrix);

  // Return to the begining of the file.
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
