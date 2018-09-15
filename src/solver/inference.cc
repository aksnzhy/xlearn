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
This file is the implementation of the Predictor class.
*/

#include "src/solver/inference.h"
#include "src/base/timer.h"
#include "src/base/format_print.h"

#include <vector>
#include <sstream>

namespace xLearn {

// Given a pre-trained model and test data, the predictor
// will return the prediction output
void Predictor::Predict() {
  std::ofstream o_file(out_file_);
  static std::vector<real_t> out;
  DMatrix* matrix = nullptr;
  reader_->Reset();
  loss_->Reset();
  for (;;) {
    index_t tmp = reader_->Samples(matrix);
    if (tmp == 0) { break; }
    if (tmp != out.size()) { out.resize(tmp); }
    loss_->Predict(matrix, *model_, out);
    if (reader_->has_label()) {
      loss_->Evalute(out, matrix->Y);
    }
    if (sigmoid_) {
      this->sigmoid(out, out);
    } else if (sign_) {
      this->sign(out, out);
    }
    for (index_t i = 0; i < out.size(); ++i) {
      o_file << out[i] << "\n";
    }
  }
  if (reader_->has_label()) {
    Color::print_info(
      StringPrintf("The test loss is: %.6f", 
        loss_->GetLoss())
    );
  }
}

// Convert output by using the sigmoid function.
void Predictor::sigmoid(std::vector<real_t>& in, 
                        std::vector<real_t>& out) {
  CHECK_EQ(in.size(), out.size());
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = 1.0 / (1.0 + exp(-in[i]));
  }
}

// Convert output to 0 and 1.
void Predictor::sign(std::vector<real_t>& in, 
                     std::vector<real_t>& out) {
  CHECK_EQ(in.size(), out.size());
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = in[i] > 0 ? 1 : 0;
  }
}

}  // namespace xLearn
