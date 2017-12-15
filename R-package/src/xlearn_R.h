#ifndef XLearn_R_H_
#define XLearn_R_H_
#define R_NO_REMAP

#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>
#include <R_ext/Random.h>

#include <src/c_api/c_api.h>
#include <src/c_api/c_api_error.h>

// Create xlearn handle
XL_DLL SEXP XLearnCreate_R(SEXP model_type, SEXP out);

// Free the xLearn handle
XL_DLL SEXP XLearnHandleFree_R(SEXP out);

// Show the model information
XL_DLL SEXP XLearnShow_R(SEXP out);

// Set file path of the training data
XL_DLL SEXP XLearnSetTrain_R(SEXP out, SEXP train_path);

// Set file path of the test data
XL_DLL SEXP XLearnSetTest_R(SEXP out, SEXP test_path);

// Set file path of the validation data
XL_DLL SEXP XLearnSetValidate_R(SEXP out, SEXP val_path);

// Start to train
XL_DLL SEXP XLearnFit_R(SEXP out, SEXP model_path);

// Cross-validation
XL_DLL SEXP XLearnCV_R(SEXP out);

// Start to predict
XL_DLL SEXP XLearnPredict_R(SEXP out, SEXP model_path, SEXP out_path);

// Set string param
XL_DLL SEXP XLearnSetStr_R(SEXP out, SEXP key, SEXP value);

// Set param
XL_DLL SEXP XLearnSetInt_R(SEXP out, SEXP key, SEXP value);

// Set float param
XL_DLL SEXP XLearnSetFloat_R(SEXP out, SEXP key, SEXP value);

// Set bool param
XL_DLL SEXP XLearnSetBool_R(SEXP out, SEXP key, SEXP value);

#endif

