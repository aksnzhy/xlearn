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

This file contains facilitlies controlling file.
*/

#ifndef XLEARN_BASE_FILE_UTIL_H_
#define XLEARN_BASE_FILE_UTIL_H_

#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>  // for remove()

#include "src/base/common.h"

//------------------------------------------------------------------------------
// Basic operations for a file
//------------------------------------------------------------------------------

// 100 KB for one line of txt data
static const uint32 kMaxLineSize = 100 * 1024;

// Check whether the file exists
inline bool FileExist(const char* filename) {
  if (access(filename, F_OK) != -1) {
    return true;
  }
  return false;
}

// Open file using fopen() and return the file pointer
// model :
//  "w" for write
//  "r" for read
inline FILE* OpenFileOrDie(const char* filename, const char* mode) {
  FILE* input_stream = fopen(filename, mode);
  if (input_stream == nullptr) {
    LOG(FATAL) << "Cannot open file: " << filename
               << " with mode: " << mode;
  }
  return input_stream;
}

// Close file using fclose()
inline void Close(FILE *file) {
  if (fclose(file) == -1) {
    LOG(FATAL) << "Error invoke fclose().";
  }
}

// Return the size (byte) of a target file
inline uint64 GetFileSize(FILE* file) {
  if (fseek(file, 0L, SEEK_END) != 0) {
    LOG(FATAL) << "Error: invoke fseek().";
  }
  uint64 total_size = ftell(file);
  if (total_size == -1) {
    LOG(FATAL) << "Error: invoke ftell().";
  }
  rewind(file);  /* Return to the head */
  return total_size;
}

// Get one line of data from the file
inline void GetLine(FILE* file, std::string& str_line) {
  CHECK_NOTNULL(file);
  static char* line = new char[kMaxLineSize];
  fgets(line, kMaxLineSize, file);
  int read_len = strlen(line);
  if (line[read_len - 1] != '\n') {
    LOG(FATAL) << "Encountered a too-long line.     \
                   Please check the data.";
  } else {
    line[read_len - 1] = '\0';
    // Handle the format in DOS and windows
    if (read_len > 1 && line[read_len - 2] == '\r') {
      line[read_len - 2] = '\0';
    }
  }
  str_line.assign(line);
}

// Write data from a buffer to disk file
// Return the size of data we have written
inline size_t WriteDataToDisk(FILE* file, const char* buf, size_t len) {
  CHECK_NOTNULL(file);
  CHECK_NOTNULL(buf);
  CHECK_GE(len, 0);
  size_t write_len = fwrite(buf, 1, len, file);
  if (write_len != len) {
    LOG(FATAL) << "Error: invoke fwrite().";
  }
  return write_len;
}

// Read data from disk file to a buffer
// Return the data size we have read
// If we reach the end of the file, return 0
inline size_t ReadDataFromDisk(FILE* file, char* buf, size_t len) {
  CHECK_NOTNULL(file);
  CHECK_NOTNULL(buf);
  CHECK_GE(len, 0);
  /* Reach the end of the file */
  if (feof(file)) {
    return 0;
  }
  size_t ret = fread(buf, 1, len, file);
  if (ret > len) {
    LOG(FATAL) << "Error: invoke fread().";
  }
  return ret;
}

// Delete target file from disk
inline void RemoveFile(const char* filename) {
  CHECK_NOTNULL(filename);
  if (remove(filename) == -1) {
    LOG(FATAL) << "Error: invoke remove().";
  }
}

//------------------------------------------------------------------------------
// Serialize or Deserialize for std::vector and std::string
//------------------------------------------------------------------------------

// Write a std::vector to disk file
template <typename T>
void WriteVectorToFile(FILE* file_ptr, const std::vector<T>& vec) {
  CHECK_NOTNULL(file_ptr);
  // We do not allow Serialize an empty vector
  CHECK(!vec.empty());
  size_t len = vec.size();
  WriteDataToDisk(file_ptr, (char*)&len, sizeof(len));
  WriteDataToDisk(file_ptr, (char*)vec.data(), sizeof(T)*len);
}

// Read a std::vector from disk file
template <typename T>
void ReadVectorFromFile(FILE* file_ptr, std::vector<T>& vec) {
  CHECK_NOTNULL(file_ptr);
  // First, read the size of vector
  size_t len = 0;
  ReadDataFromDisk(file_ptr, (char*)(&len), sizeof(len));
  CHECK_GT(len, 0);
  vec.resize(len);
  ReadDataFromDisk(file_ptr, (char*)vec.data(), sizeof(T)*len);
}

// Write string to disk file
inline void WriteStringToFile(FILE* file_ptr, const std::string& str) {
  CHECK_NOTNULL(file_ptr);
  // We do not allow Serialize an empty string
  CHECK(!str.empty());
  size_t len = str.size();
  WriteDataToDisk(file_ptr, (char*)&len, sizeof(len));
  WriteDataToDisk(file_ptr, (char*)str.data(), len);
}

// Read string from disk file
inline void ReadStringFromFile(FILE* file_ptr, std::string& str) {
  CHECK_NOTNULL(file_ptr);
  // First, read the size of vector
  size_t len = 0;
  ReadDataFromDisk(file_ptr, (char*)(&len), sizeof(len));
  CHECK_GT(len, 0);
  str.resize(len);
  ReadDataFromDisk(file_ptr, (char*)str.data(), len);
}

#endif  // XLEARN_BASE_FILE_UTIL_H_
