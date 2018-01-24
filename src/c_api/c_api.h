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

This file define the C API of xLearn, used for interfacing 
to other languages.
*/

#ifndef XLEARN_C_API_C_API_H_
#define XLEARN_C_API_C_API_H_

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/solver/solver.h"

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

// Set data matrix for the training set
XL_DLL int XLearnSetTrainDMatrix(XL *out, XL *train_matrix);

// Set file path of the test data
XL_DLL int XLearnSetTest(XL *out, const char *test_path);

// Set data matrix for the test set
XL_DLL int XLearnSetTestDMatrix(XL *out, XL *test_matrix);

// Set file path of the validation data
XL_DLL int XLearnSetValidate(XL *out, const char *val_path);

// Set data matrix for the validation set
XL_DLL int XLearnSetValidateDMatrix(XL *out, XL *val_matrix);

// Set file path of the txt model data
XL_DLL int XLearnSetTXTModel(XL *out, const char *model_path);

// Start to train
XL_DLL int XLearnFit(XL *out, const char *model_path);

// Cross-validation
XL_DLL int XLearnCV(XL *out);

// Start to predict
XL_DLL int XLearnPredict(XL *out, const char *model_path, const char *out_path);

// Set string param
XL_DLL int XLearnSetStr(XL *out, const char *key, const char *value);

// Set int param
XL_DLL int XLearnSetInt(XL *out, const char *key, const int value);

// Set float param
XL_DLL int XLearnSetFloat(XL *out, const char *key, const float value);

// Set bool param
XL_DLL int XLearnSetBool(XL *out, const char *key, const bool value);

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