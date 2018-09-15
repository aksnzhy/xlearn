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
This file is the implementation of StrSimilar class.
*/

#include <algorithm>

#include "src/base/levenshtein_distance.h"

#define min(a,b) ((a<b)?a:b)

// Find str in string list.
// Return true if we can find str in target string list.
bool StrSimilar::Find(const std::string& str,
                      const std::vector<std::string>& list) {
  CHECK(!str.empty());
  CHECK(!list.empty());
  std::vector<std::string>::const_iterator it;
  it = std::find(list.begin(), list.end(), str);
  if (it != list.end()) {
    return true;
  }
  return false;
}

// Find the most similar string from string list.
// Return the minimal levenshtein distance.
int StrSimilar::FindSimilar(const std::string& str,
                            const std::vector<std::string>& list,
                            std::string& result) {
  CHECK(!str.empty());
  CHECK(!list.empty());
  int min_dis = kInt32Max;
  for (int i = 0; i < list.size(); ++i) {
    int dis = ldistance(str, list[i]);
    if (dis < min_dis) {
      min_dis = dis;
      result = list[i];
    }
  }
  return min_dis;
}

// Calculate Levenshtein distance by using
// dynamic programing (DP).
int StrSimilar::ldistance(const std::string& source,
                          const std::string& target) {
  CHECK(!source.empty());
  CHECK(!target.empty());
  //step 1
  int n = source.length();
  int m = target.length();
  if (m == 0) return n;
  if (n == 0) return m;
  //Construct a matrix
  typedef std::vector<std::vector<int> >  Tmatrix;
  Tmatrix matrix(n + 1);
  for (int i = 0; i <= n; i++)  { matrix[i].resize(m + 1); }
  //step 2 Initialize
  for (int i = 1; i <= n; i++) { matrix[i][0] = i; }
  for (int i = 1; i <= m; i++) { matrix[0][i] = i; }
  //step 3
  for (int i = 1; i <= n; i++) {
    const char si = source[i - 1];
    //step 4
    for (int j = 1; j <= m; j++) {
      const char dj = target[j - 1];
      //step 5
      int cost;
      if (si == dj) { cost = 0; }
      else { cost = 1; }
      //step 6
      const int above = matrix[i - 1][j] + 1;
      const int left = matrix[i][j - 1] + 1;
      const int diag = matrix[i - 1][j - 1] + cost;
      matrix[i][j] = min(above, min(left, diag));
     }
  }
  //step 7
  return matrix[n][m];
}
