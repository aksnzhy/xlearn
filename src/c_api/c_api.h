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

typedef void *SolverHandle;
typedef void *HyperParamHandle;

// Create a linear model
XL_DLL int CreateLinear();

// Create a factorization machine
XL_DLL int CreateFM();

// Create a field-aware factorization machine
XL_DLL int CreateFFM();

// Set file path of the training data
XL_DLL int SetTrain();

// Set file path of the test data
XL_DLL int SetTest();

// Set file path of the validation data
XL_DLL int SetValidation();

// Start to train
XL_DLL int Train();

// Start to predict
XL_DLL int Predict();

#endif  // XLEARN_C_API_C_API_H_