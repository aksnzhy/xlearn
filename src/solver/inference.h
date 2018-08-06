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
This file defines the Predictor class.
*/

#ifndef XLEARN_SOLVER_INFERENCE_H_
#define XLEARN_SOLVER_INFERENCE_H_

#include <string>

#include "src/base/common.h"
#include "src/data/data_structure.h"
#include "src/data/model_parameters.h"
#include "src/reader/reader.h"
#include "src/loss/loss.h"

namespace xLearn {

//------------------------------------------------------------------------------
// Given a pre-trained model and test data, the predictor
// will return the prediction output
//------------------------------------------------------------------------------
class Predictor {
 public:
  // Constructor and Desstructor
  Predictor() { }
  ~Predictor() { }

  // Invoke this function before we use this class
  void Initialize(Reader* reader,
                  Model* model,
                  Loss* loss,
                  const std::string& out,
                  bool sign = false,
                  bool sigmoid = false) {
    CHECK_NOTNULL(reader);
    CHECK_NOTNULL(model);
    CHECK_NOTNULL(loss);
    CHECK_NE(out.empty(), true);
    reader_ = reader;
    model_ = model;
    loss_ = loss;
    out_file_ = out;
    sign_ = sign;
    sigmoid_ = sigmoid;
  }

  // The core function
  void Predict();

 protected:
  Reader* reader_;
  Model* model_;
  Loss* loss_;
  std::string out_file_;
  bool sign_;
  bool sigmoid_;

  // Convert output by using the sigmoid function.
  void sigmoid(std::vector<real_t>& in, 
               std::vector<real_t>& out);

  // Convert output to 0 and 1.
  void sign(std::vector<real_t>& in, 
            std::vector<real_t>& out);

 private:
  DISALLOW_COPY_AND_ASSIGN(Predictor);
};  // class Predictor

}  // namespace xLearn

#endif // XLEARN_SOLVER_INFERENCE_H_
