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
#include <iostream>

#include <string.h>

#include "src/c_api/c_api.h"
#include "src/c_api/c_api_error.h"
#include "src/base/format_print.h"
#include "src/base/timer.h"

// Say hello to user
XL_DLL int XLearnHello() {
  API_BEGIN();
  std::string logo = 
"----------------------------------------------------------------------------------------------\n"
                    "           _\n"
                    "          | |\n"
                    "     __  _| |     ___  __ _ _ __ _ __\n"
                    "     \\ \\/ / |    / _ \\/ _` | '__| '_ \\ \n"
                    "      >  <| |___|  __/ (_| | |  | | | |\n"
                    "     /_/\\_\\_____/\\___|\\__,_|_|  |_| |_|\n\n"
                    "        xLearn   -- 0.20 Version --\n"
"----------------------------------------------------------------------------------------------\n"
"\n";
  Color::Modifier green(Color::FG_GREEN);
  Color::Modifier def(Color::FG_DEFAULT);
  Color::Modifier bold(Color::BOLD);
  Color::Modifier reset(Color::RESET);
  std::cout << green << bold << logo << def << reset;
  API_END();
}

// Create xlearn handle
XL_DLL int XLearnCreate(const char *model_type, XL *out) {
  API_BEGIN();
  XLearn* xl = new XLearn;
  xl->GetHyperParam().score_func = std::string(model_type);
  *out = xl;
  API_END();
}

// Free the xLearn handle
XL_DLL int XLearnHandleFree(XL *out) {
  API_BEGIN();
  // For now, we do nothing here
  API_END();
}

// Show the mode information
XL_DLL int XLearnShow(XL *out) {
  API_BEGIN()
  XLearn* xl = reinterpret_cast<XLearn*>(*out);
  printf("Info: \n Model: %s\n Loss: %s\n", 
  	xl->GetHyperParam().score_func.c_str(),
  	xl->GetHyperParam().loss_func.c_str());
  API_END()
}

// Set file path of the training data
XL_DLL int XLearnSetTrain(XL *out, const char *train_path) {
  API_BEGIN();
  XLearn* xl = reinterpret_cast<XLearn*>(*out);
  xl->GetHyperParam().train_set_file = std::string(train_path);
  API_END();
}

// Set data matrix for the training set
XL_DLL int XLearnSetTrainDMatrix(XL *out, XL *train_matrix) {
  API_BEGIN();
  //XLearn* xl = reinterpret_cast<XLearn*>(*out);

  API_END();
}

// Set file path of the test data
XL_DLL int XLearnSetTest(XL *out, const char *test_path) {
  API_BEGIN();
  XLearn* xl = reinterpret_cast<XLearn*>(*out);
  xl->GetHyperParam().test_set_file = std::string(test_path);
  API_END();
}

// Set data matrix for the test set
XL_DLL int XLearnSetTestDMatrix(XL *out, XL *test_matrix) {
  API_BEGIN();
  //XLearn* xl = reinterpret_cast<XLearn*>(*out);

  API_END();
}

// Set file path of the validation data
XL_DLL int XLearnSetValidate(XL *out, const char *val_path) {
  API_BEGIN();
  XLearn* xl = reinterpret_cast<XLearn*>(*out);
  xl->GetHyperParam().validate_set_file = std::string(val_path);
  API_END();
}

// Set data matrix for the validation set
XL_DLL int XLearnSetValidateDMatrix(XL *out, XL *val_matrix) {
  API_BEGIN();
  //XLearn* xl = reinterpret_cast<XLearn*>(*out);

  API_END();
}

// Set file path of the txt model data
XL_DLL int XLearnSetTXTModel(XL *out, const char *model_path) {
  API_BEGIN();
  XLearn* xl = reinterpret_cast<XLearn*>(*out);
  xl->GetHyperParam().txt_model_file = std::string(model_path);
  API_END();
}

// Start to train
XL_DLL int XLearnFit(XL *out, const char *model_path) {
  API_BEGIN();
  Timer timer;
  timer.tic();
  XLearn* xl = reinterpret_cast<XLearn*>(*out);
  xl->GetHyperParam().model_file = std::string(model_path);
  xl->GetHyperParam().is_train = true;
  xl->GetSolver().Initialize(xl->GetHyperParam());
  xl->GetSolver().StartWork();
  xl->GetSolver().Clear();
  print_info(
    StringPrintf("Total time cost: %.2f (sec)", 
    timer.toc()), true);
  API_END();
}

// Cross-validation
XL_DLL int XLearnCV(XL *out) {
  API_BEGIN();
  Timer timer;
  timer.tic();
  XLearn* xl = reinterpret_cast<XLearn*>(*out);
  xl->GetHyperParam().cross_validation = true;
  xl->GetHyperParam().is_train = true;
  xl->GetSolver().Initialize(xl->GetHyperParam());
  xl->GetSolver().StartWork();
  xl->GetSolver().Clear();
  xl->GetHyperParam().cross_validation = false;
  print_info(
    StringPrintf("Total time cost: %.2f (sec)", 
    timer.toc()), true);
  API_END();
}

// Start to predict
XL_DLL int XLearnPredict(XL *out, const char *model_path, const char *out_path) {
  API_BEGIN();
  Timer timer;
  timer.tic();
  XLearn* xl = reinterpret_cast<XLearn*>(*out);
  xl->GetHyperParam().model_file = std::string(model_path);
  xl->GetHyperParam().output_file = std::string(out_path);
  xl->GetHyperParam().is_train = false;
  xl->GetSolver().Initialize(xl->GetHyperParam());
  xl->GetSolver().SetPredict();
  xl->GetSolver().StartWork();
  xl->GetSolver().Clear();
  print_info(
    StringPrintf("Total time cost: %.2f (sec)", 
    timer.toc()), true);
  API_END();
}

// Set string param
XL_DLL int XLearnSetStr(XL *out, const char *key, const char *value) {
  API_BEGIN();
  XLearn* xl = reinterpret_cast<XLearn*>(*out);
  if (strcmp(key, "task") == 0) {
  	if (strcmp(value, "binary") == 0) {
  	  xl->GetHyperParam().loss_func = std::string("cross-entropy");
  	} else if (strcmp(value, "reg") == 0) {
  	  xl->GetHyperParam().loss_func = std::string("squared");
  	} else {
      xl->GetHyperParam().loss_func = std::string("unknow");
    }
  } else if (strcmp(key, "metric") == 0) {
  	xl->GetHyperParam().metric = std::string(value);
  } else if (strcmp(key, "log") == 0) {
  	xl->GetHyperParam().log_file = std::string(value);
  } else if (strcmp(key, "loss") == 0) {
  	xl->GetHyperParam().loss_func = std::string(value);
  } else if (strcmp(key, "opt") == 0) {
    xl->GetHyperParam().opt_type = std::string(value);
  }
  API_END();
}

// Set int param
XL_DLL int XLearnSetInt(XL *out, const char *key, const int value) {
  API_BEGIN();
  XLearn* xl = reinterpret_cast<XLearn*>(*out);
  if (strcmp(key, "k") == 0) {
  	xl->GetHyperParam().num_K = value;
  } else if (strcmp(key, "epoch") == 0) {
  	xl->GetHyperParam().num_epoch = value;
  } else if (strcmp(key, "fold") == 0) {
  	xl->GetHyperParam().num_folds = value;
  } else if (strcmp(key, "block_size") == 0) {
  	xl->GetHyperParam().block_size = value;
  } else if (strcmp(key, "nthread") == 0) {
    xl->GetHyperParam().thread_number = value;
  }
  API_END();
}

// Set float param
XL_DLL int XLearnSetFloat(XL *out, const char *key, const float value) {
  API_BEGIN();
  XLearn* xl = reinterpret_cast<XLearn*>(*out);
  if (strcmp(key, "lr") == 0) {
  	xl->GetHyperParam().learning_rate = value;
  } else if (strcmp(key, "lambda") == 0) {
  	xl->GetHyperParam().regu_lambda = value;
  } else if (strcmp(key, "init") == 0) {
  	xl->GetHyperParam().model_scale = value;
  } else if (strcmp(key, "alpha") == 0) {
    xl->GetHyperParam().alpha = value;
  } else if (strcmp(key, "beta") == 0) {
    xl->GetHyperParam().beta = value;
  } else if (strcmp(key, "lambda_1") == 0) {
    xl->GetHyperParam().lambda_1 = value;
  } else if (strcmp(key, "lambda_2") == 0) {
    xl->GetHyperParam().lambda_2 = value;
  }
  API_END();
}

// Set bool param
XL_DLL int XLearnSetBool(XL *out, const char *key, const bool value) {
  API_BEGIN();
  XLearn* xl = reinterpret_cast<XLearn*>(*out);
  if (strcmp(key, "on_disk") == 0) {
  	xl->GetHyperParam().on_disk = value;
  } else if (strcmp(key, "quiet") == 0) {
  	xl->GetHyperParam().quiet = value;
  } else if (strcmp(key, "norm") == 0) {
  	xl->GetHyperParam().norm = value;
  } else if (strcmp(key, "lock_free") == 0) {
  	xl->GetHyperParam().lock_free = value;
  } else if (strcmp(key, "early_stop") == 0) {
  	xl->GetHyperParam().early_stop = value;
  } else if (strcmp(key, "sign") == 0) {
  	xl->GetHyperParam().sign = value;
  } else if (strcmp(key, "sigmoid") == 0) {
  	xl->GetHyperParam().sigmoid = value;
  }
  API_END();
}
