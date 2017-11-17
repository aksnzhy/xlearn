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

This file is the implementation of C API for xLearn.
*/

#include <string>

#include "src/c_api/c_api.h"

// Create xlearn handle
XL_DLL int XLearnCreate(const char *model_type,
	                    XLearnHandle *out) {
  API_BEGIN();
  XLearn* xl = new xLearn();
  xl->hyper_param.score_func = std::string(model_type);
  *out = xl;
  API_END();
}

// Free the xLearn handle
XL_DLL int XLearnHandleFree(XLearnHandle *out) {
  API_BEGIN();
  // For now, we do nothing here
  API_END();
}

// Set file path of the training data
XL_DLL int XLearnSetTrain(XLearnHandle *out,
	                      const char *train_path) {
  API_BEGIN();
  out->hyper_param.train_set_file = std::string(train_path);
  API_END();
}

// Set file path of the test data
XL_DLL int XLearnSetTest(XLearnHandle *out,
	                     const char *test_path) {
  API_BEGIN();
  out->hyper_param.test_set_file = std::string(test_path);
  API_END();
}

// Set file path of the validation data
XL_DLL int XLearnSetValidate(XLearnHandle *out,
	                         const char *val_path) {
  API_BEGIN();
  out->hyper_param.validate_set_file = std::string(val_path);
  API_END();
}

// Start to train
XL_DLL int XLearnFit(XLearnHandle *out,
	                 const char *model_path) {
  API_BEGIN();
  out->hyper_param.model_file = std::string(model_path);
  out->solver.SetTrain();
  out->solver.Initialize(out->hyper_param);
  out->solver.StartWork();
  out->solver.FinalizeWork();
  API_END();
}

// Cross-validation
XL_DLL int XLearnCV(XLearnHandle *out) {
  API_BEGIN();
  out->hyper_param.cross_validation = true;
  out->solver.SetTrain();
  out->solver.Initialize(out->hyper_param);
  out->solver.StartWork();
  out->solver.FinalizeWork();
  out->hyper_param.cross_validation = false;
  API_END();
}

// Start to predict
XL_DLL int XLearnPredict(XLearnHandle *out,
	                     const char *model_path,
	                     const char *out_path) {
  API_BEGIN();
  out->hyper_param.model_file = std::string(model_path);
  out->hyper_param.output_file = std::string(out);
  out->solver.SetPredict();
  out->solver.Initialize(out->hyper_param);
  out->solver.StartWork();
  out->solver.FinalizeWork();
  API_END();
}

// Set string param
XL_DLL int XLearnSetStr(XLearnHandle *out,
	                    const char *key,
	                    const char *value) {
  API_BEGIN();
  if (key == "task") {
  	if (value == "binary") {
  	  out->hyper_param.loss_func = std::string("cross-entropy");
  	} else if (value == "reg") {
  	  out->hyper_param.loss_func = std::string("squared");
  	}
  } else if (key == "metric") {
  	out->hyper_param.metric = std::string(value);
  } else if (key == "log") {
  	out->hyper_param.log_file = std::string(value);
  }
  API_END();
}

// Set int param
XL_DLL int XLearnSetInt(XLearnHandle *out,
	                    const char *key,
	                    const int value) {
  API_BEGIN();
  if (key == "k") {
  	out->hyper_param.num_K = value;
  } else if (key == "epoch") {
  	out->hyper_param.num_epoch = value;
  } else if (key == "fold") {
  	out->hyper_param.num_folds = value;
  }
  API_END();
}

// Set float param
XL_DLL int XLearnSetFloat(XLearnHandle *out,
	                      const char *key,
	                      const float value) {
  API_BEGIN();
  if (key == "lr") {
  	out->hyper_param.learning_rate = value;
  } else if (key == "lambda") {
  	out->hyper_param.regu_lambda = value;
  } else if (key == "init") {
  	out->hyper_param.model_scale = value;
  }
  API_END();
}

// Set bool param
XL_DLL int XLearnSetBool(XLearnHandle *out,
	                     const char *key,
	                     const float value) {
  API_BEGIN();
  if (key == "on_disk") {
  	out->hyper_param.on_disk = value;
  } else if (key == "quiet") {
  	out->hyper_param.quiet = value;
  } else if (key == "norm") {
  	out->hyper_param.norm = value;
  } else if (key == "lock_free") {
  	out->hyper_param.lock_free = value;
  } else if (key == "early_stop") {
  	out->hyper_param.early_stop = value;
  } else if (key == "sign") {
  	out->sign = value;
  } else if (key == "sigmoid") {
  	out->sigmoid = value;
  }
  API_END();
}