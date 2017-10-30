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
This file is the implementation of the Predictor class.
*/

#include "src/solver/inference.h"

#include <vector>
#include <sstream>

namespace xLearn {

// Given a pre-trained model and test data, the predictor
// will return the prediction output
void Predictor::Predict() {
  std::ofstream o_file(out_file_);
  static std::vector<real_t> out;
  DMatrix* matrix = nullptr;
  index_t total_count = 0;
  real_t loss_val = 0;
  reader_->Reset();
  for (;;) {
    index_t tmp = reader_->Samples(matrix);
    if (tmp == 0) { break; }
    if (tmp != out.size()) { out.resize(tmp); }
    total_count += tmp;
    loss_->Predict(matrix, *model_, out);
    if (reader_->has_label()) {
      loss_val += loss_->Evalute(out, matrix->Y);
    }
    if (loss_->loss_type().compare("corss-entropy")) {
      loss_->Sigmoid(out, out);
    }
    for (index_t i = 0; i < out.size(); ++i) {
      o_file << out[i] << "\n";
    }
  }
  if (reader_->has_label()) {
    loss_val /= total_count;
    printf("The tesl loss is: %.6f\n", loss_val);
  }
}

}  // namespace xLearn
