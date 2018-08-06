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
This file define the C API of xLearn, which is used 
for interfacing to other languages.
*/

#ifndef XLEARN_C_API_C_API_H_
#define XLEARN_C_API_C_API_H_

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/solver/solver.h"

#include <string>

#ifdef __cplusplus
#define XL_EXTERN_C extern "C"
#include <cstdio>
#else
#define XL_EXTERN_C
#include <stdio.h>
#include <stdint.h>
#endif

#if defined(__MSC_VER) || defined(__WIN32)
#define XL_DLL XL_EXTERN_C __declspec(dllexport)
#else
#define XL_DLL XL_EXTERN_C
#endif

/* Handle to xlearn */
typedef void* XL;

// Say hello to user
XL_DLL int XLearnHello();

// Create xlearn handle
XL_DLL int XLearnCreate(const char *model_type, XL *out);

// Free the xLearn handle
XL_DLL int XLearnHandleFree(XL *out);

// Show the model information
XL_DLL int XLearnShow(XL *out);

// Set file path of the training data
XL_DLL int XLearnSetTrain(XL *out, const char *train_path);

// Get file path of th training data
XL_DLL int XLearnGetTrain(XL *out, std::string& train_path);

// Set file path of the test data
XL_DLL int XLearnSetTest(XL *out, const char *test_path);

// Get file path of the test data
XL_DLL int XLearnGetTest(XL *out, std::string& test_path);

// Set file path of the validation data
XL_DLL int XLearnSetValidate(XL *out, const char *val_path);

// Get file path of the validation data
XL_DLL int XLearnGetValidate(XL *out, std::string& val_path);

// Set file path of the txt model
XL_DLL int XLearnSetTXTModel(XL *out, const char *model_path);

// Get file path of the txt model
XL_DLL int XLearnGetTXTModel(XL *out, std::string& model_path);

// Start to train
XL_DLL int XLearnFit(XL *out, const char *model_path);

// Cross-validation
XL_DLL int XLearnCV(XL *out);

// Start to predict
XL_DLL int XLearnPredict(XL *out, const char *model_path, const char *out_path);

// Set string param
XL_DLL int XLearnSetStr(XL *out, const char *key, const char *value);

// Get string params
XL_DLL int XLearnGetStr(XL* out, const char *key, std::string& value);

// Set int param
XL_DLL int XLearnSetInt(XL *out, const char *key, const int value);

// Get int param
XL_DLL int XLearnGetInt(XL *out, const char *key, int *value);

// Set float param
XL_DLL int XLearnSetFloat(XL *out, const char *key, const float value);

// Get float param
XL_DLL int XLearnGetFloat(XL *out, const char *key, float *value);

// Set bool param
XL_DLL int XLearnSetBool(XL *out, const char *key, const bool value);

// Get bool param
XL_DLL int XLearnGetBool(XL *out, const char *key, bool *value);

// This is the entry class used by c_api.
class XLearn {
 public:
  // Constructor and Destructor
  XLearn() {}
  ~XLearn() {}

  // Get funtions
  inline xLearn::HyperParam& GetHyperParam() { 
  	return hyper_param; 
  }
  
  inline xLearn::Solver& GetSolver() { 
  	return solver; 
  }

 protected:
   xLearn::HyperParam hyper_param;
   xLearn::Solver solver;

 private:
  DISALLOW_COPY_AND_ASSIGN(XLearn);
};

#endif  // XLEARN_C_API_C_API_H_