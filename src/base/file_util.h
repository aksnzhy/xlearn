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

This file contains facilitlies controlling file I/O.
*/

#ifndef XLEARN_BASE_FILE_UTIL_H_
#define XLEARN_BASE_FILE_UTIL_H_

#include <stdio.h> // for remove()

#include "src/base/common.h"

//------------------------------------------------------------------------------
// Basic operations for a file
//------------------------------------------------------------------------------

// Open file using fopen.
// Return the file pointer.
inline FILE* OpenFileOrDie(const char* filename, const char* mode) {
  FILE* input_stream = fopen(filename, mode);
  if (input_stream == nullptr) {
    LOG(FATAL) << "Cannot open file: " << filename
               << " with mode: " << mode;
  }
  return input_stream;
}

// Close file using fclose.
inline void Close(FILE *file) {
  if (fclose(file) == -1) {
    LOG(FATAL) << "Error invoke fclose().";
  }
}

// Get the size of a targte file.
// Return the size. Note that here we need use uint64 for large files.
inline uint64 GetFileSize(FILE* file) {
  if (fseek(file, 0L, SEEK_END) != 0) {
    LOG(FATAL) << "Error: invoke fseek().";
  }
  uint64 total_size = ftell(file);
  if (total_size == -1) {
    LOG(FATAL) << "Error: invoke ftell().";
  }
  rewind(file);
  return total_size;
}

// Write data from a buffer to target file.
// Return the size we write.
inline size_t WriteDataToDisk(FILE* file, const char* buf, size_t len) {
  size_t write_len = fwrite(buf, 1, len, file);
  if (write_len != len) {
    LOG(FATAL) << "Error: invoke fwrite().";
  }
  return write_len;
}

// Read data from target file to a buffer.
// Return the data size we read.
// If reach the end of the file, return 0.
inline size_t ReadDataFromDisk(FILE* file, char* buf, size_t max_len) {
  CHECK_GT(max_len, 0);
  if (feof(file)) {
    return 0;
  }
  size_t read_len = fread(buf, 1, max_len, file);
  return read_len;
}

// Delete target file from disk.
inline void RemoveFile(const char* filename) {
  if (remove(filename) == -1) {
    LOG(FATAL) << "Error: invoke remove().";
  }
}

//------------------------------------------------------------------------------
// Serialize and Deserialize a vector to disk file.
//------------------------------------------------------------------------------

// Serialize a vector to a buffer. Return the buffer size.
template <typename T>
size_t serialize_vector(const std::vector<T>& vec, char* &buf) {
  static size_t elem_size = sizeof(T);
  static size_t len_size = sizeof(size_t);
  CHECK_GT(vec.size(), 0);
  // The first element is the length of vector
  size_t buffer_size = elem_size * vec.size() + len_size;
  buf = new char[buffer_size];
  size_t vec_len = vec.size();
  memcpy(buf, reinterpret_cast<char*>(&vec_len), len_size);
  // The vector elements
  size_t total_size = len_size;
  T* p = (T*)vec.data();
  for (size_t i = 0; i < vec.size(); ++i) {
    memcpy(buf + total_size,
           reinterpret_cast<char*>(p+i),
           elem_size);
    total_size += elem_size;
  }

  return total_size;
}

// Deserialize a vector from a buffer.
template <typename T>
void deserialize_vector(char* buf, size_t buf_len, std::vector<T>& vec) {
  static size_t elem_size = sizeof(T);
  static size_t len_size = sizeof(size_t);
  CHECK_NOTNULL(buf);
  // Parse the first length
  size_t vec_len = 0;
  memcpy(&vec_len, buf, len_size);
  CHECK_GT(vec_len, 0);
  vec.resize(vec_len);
  // Parse the last elements
  size_t index = 0;
  for (size_t i = len_size; i < buf_len; i += elem_size) {
    T* value = reinterpret_cast<T*>(buf + i);
    vec[index++] = *value;
  }
  CHECK_EQ(index, vec.size());
}

// Write a vector of data to disk file.
template <typename T>
void WriteVectorToFile(FILE* file_ptr, const std::vector<T>& vec) {
  char* buf = nullptr;
  size_t buf_len = serialize_vector(vec, buf);
  CHECK_EQ(WriteDataToDisk(file_ptr, buf, buf_len), buf_len);
  delete [] buf;
}

// Read a vector of data from disk file.
template <typename T>
void ReadVectorFromFile(FILE* file_ptr, std::vector<T>& vec) {
  static size_t len_size = sizeof(size_t);
  // Read the size of vector
  size_t vec_len = 0;
  CHECK_GT(fread(reinterpret_cast<char*>(&vec_len),
                 1,
                 len_size,
                 file_ptr), 0);
  // Read the last elements
  size_t buf_len = sizeof(T) * vec_len + len_size;
  char* buf = new char[buf_len];
  memcpy(buf, reinterpret_cast<char*>(&vec_len), len_size);
  ReadDataFromDisk(file_ptr, buf+len_size, buf_len-len_size);
  deserialize_vector(buf, buf_len, vec);
  delete [] buf;
}

#endif // XLEARN_BASE_FILE_UTIL_H_
