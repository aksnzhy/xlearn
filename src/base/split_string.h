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
This file provides StringSplit utilities.
*/

#ifndef XLEARN_BASE_SPLIT_STRING_H_
#define XLEARN_BASE_SPLIT_STRING_H_

#include <set>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// Subdivide string |full| into substrings according to delimitors
// given in |delim|.  |delim| should pointing to a string including
// one or more characters.  Each character is considerred a possible
// delimitor. For example:
//
//   vector<string> substrings;
//   SplitStringUsing("apple orange\tbanana", "\t ", &substrings);
//
// results in three substrings:
//
//   substrings.size() == 3
//   substrings[0] == "apple"
//   substrings[1] == "orange"
//   substrings[2] == "banana"
//------------------------------------------------------------------------------

void SplitStringUsing(const std::string& full,
                      const char* delim,
                      std::vector<std::string>* result);

// This function has the same semnatic as SplitStringUsing.  Results
// are saved in an STL set container.
void SplitStringToSetUsing(const std::string& full,
                           const char* delim,
                           std::set<std::string>* result);

template <typename T>
struct simple_insert_iterator {
  explicit simple_insert_iterator(T* t) : t_(t) { }

  simple_insert_iterator<T>& operator=(const typename T::value_type& value) {
    t_->insert(value);
    return *this;
  }

  simple_insert_iterator<T>& operator*() { return *this; }
  simple_insert_iterator<T>& operator++() { return *this; }
  simple_insert_iterator<T>& operator++(int placeholder) { return *this; }

  T* t_;
};

template <typename T>
struct back_insert_iterator {
  explicit back_insert_iterator(T& t) : t_(t) {}

  back_insert_iterator<T>& operator=(const typename T::value_type& value) {
    t_.push_back(value);
    return *this;
  }

  back_insert_iterator<T>& operator*() { return *this; }
  back_insert_iterator<T>& operator++() { return *this; }
  back_insert_iterator<T> operator++(int placeholder) { return *this; }

  T& t_;
};

template <typename StringType, typename ITR>
static inline
void SplitStringToIteratorUsing(const StringType& full,
                                const char* delim,
                                ITR& result) {
  // Optimize the common case where delim is a single character.
  if (delim[0] != '\0' && delim[1] == '\0') {
    char c = delim[0];
    const char* p = full.data();
    const char* end = p + full.size();
    while (p != end) {
      if (*p == c) {
        ++p;
      } else {
        const char* start = p;
        while (++p != end && *p != c) {
          // Skip to the next occurence of the delimiter.
        }
        *result++ = StringType(start, p - start);
      }
    }
    return;
  }

  std::string::size_type begin_index, end_index;
  begin_index = full.find_first_not_of(delim);
  while (begin_index != std::string::npos) {
    end_index = full.find_first_of(delim, begin_index);
    if (end_index == std::string::npos) {
      *result++ = full.substr(begin_index);
      return;
    }
    *result++ = full.substr(begin_index, (end_index - begin_index));
    begin_index = full.find_first_not_of(delim, end_index);
  }
}

#endif   // XLEARN_BASE_SPLIT_STRING_H_
