#include "./xlearn_R.h"

/*!
* \brief macro to annotate begin of api
*/
#define R_API_BEGIN()                           \
GetRNGstate();                                  \
try {
/*!
    * \brief macro to annotate end of api
    */
#define R_API_END()                             \
} catch(...) {                                  \
    PutRNGstate();                              \
    Rf_error("Error");                          \
}                                               \
PutRNGstate();

/*!
* \brief macro to check the call.
*/
#define CHECK_CALL(x)                           \
if ((x) != 0) {                                 \
    Rf_error(XLearnGetLastError());             \
}

// Say hello to user
SEXP XLearnHello_R() {
    R_API_BEGIN();
    CHECK_CALL(XLearnHello());
    R_API_END();
}

void _XLearnFinalizer(SEXP ext) {
    R_API_BEGIN();
    if (R_ExternalPtrAddr(ext) == NULL) return;
    void *r_exptr=R_ExternalPtrAddr(ext);
    CHECK_CALL(XLearnHandleFree(&r_exptr));
    R_ClearExternalPtr(ext);
    R_API_END();
}

// Create xlearn handle
SEXP XLearnCreate_R(SEXP model_type) {
    SEXP ret;
    XL out;
    R_API_BEGIN();
    CHECK_CALL(XLearnCreate(CHAR(Rf_asChar(model_type)), &out));
    ret = PROTECT(R_MakeExternalPtr(out, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ret, _XLearnFinalizer, TRUE);
    R_API_END();
    UNPROTECT(1);
    return ret;
}

// Show the model information
SEXP XLearnShow_R(SEXP out) {
    R_API_BEGIN();
    void *r_exptr=R_ExternalPtrAddr(out);
    CHECK_CALL(XLearnShow(&r_exptr));
    R_API_END();
}

// Set file path of the training data
SEXP XLearnSetTrain_R(SEXP out, SEXP train_path) {
    R_API_BEGIN();
    void *r_exptr=R_ExternalPtrAddr(out);
    CHECK_CALL(XLearnSetTrain(&r_exptr,
                              CHAR(Rf_asChar(train_path))));
    R_API_END();
}

// Set file path of the test data
SEXP XLearnSetTest_R(SEXP out, SEXP test_path) {
    R_API_BEGIN();
    void *r_exptr=R_ExternalPtrAddr(out);
    CHECK_CALL(XLearnSetTest(&r_exptr,
                             CHAR(Rf_asChar(test_path))));
    R_API_END();
}

// Set file path of the validation data
SEXP XLearnSetValidate_R(SEXP out, SEXP val_path) {
    R_API_BEGIN();
    void *r_exptr=R_ExternalPtrAddr(out);
    CHECK_CALL(XLearnSetValidate(&r_exptr,
                                 CHAR(Rf_asChar(val_path))));
    R_API_END();
}

// Start to train
SEXP XLearnFit_R(SEXP out, SEXP model_path) {
    R_API_BEGIN();
    void *r_exptr=R_ExternalPtrAddr(out);
    CHECK_CALL(XLearnFit(&r_exptr,
                         CHAR(Rf_asChar(model_path))));
    R_API_END();
}

// Cross-validation
SEXP XLearnCV_R(SEXP out) {
    R_API_BEGIN();
    void *r_exptr=R_ExternalPtrAddr(out);
    CHECK_CALL(XLearnCV(&r_exptr));
    R_API_END();
}

// Start to predict
SEXP XLearnPredict_R(SEXP out, SEXP model_path, SEXP out_path) {
    R_API_BEGIN();
    void *r_exptr=R_ExternalPtrAddr(out);
    CHECK_CALL(XLearnPredict(&r_exptr,
                             CHAR(Rf_asChar(model_path)),
                             CHAR(Rf_asChar(out_path))));
    R_API_END();
}

// Set string param
SEXP XLearnSetStr_R(SEXP out, SEXP key, SEXP value) {
    R_API_BEGIN();
    void *r_exptr=R_ExternalPtrAddr(out);
    CHECK_CALL(XLearnSetStr(&r_exptr,
                            CHAR(Rf_asChar(key)),
                            CHAR(Rf_asChar(value))));
    R_API_END();
}

// Set int param
SEXP XLearnSetInt_R(SEXP out, SEXP key, SEXP value) {
    R_API_BEGIN();
    void *r_exptr=R_ExternalPtrAddr(out);
    CHECK_CALL(XLearnSetInt(&r_exptr,
                            CHAR(Rf_asChar(key)),
                            Rf_asInteger(value)));
    R_API_END();
}

// Set float param
SEXP XLearnSetFloat_R(SEXP out, SEXP key, SEXP value) {
    R_API_BEGIN();
    void *r_exptr=R_ExternalPtrAddr(out);
    CHECK_CALL(XLearnSetFloat(&r_exptr,
                              CHAR(Rf_asChar(key)),
                              Rf_asReal(value)));
    R_API_END();
}

// Set bool param
SEXP XLearnSetBool_R(SEXP out, SEXP key, SEXP value) {
    R_API_BEGIN();
    void *r_exptr=R_ExternalPtrAddr(out);
    CHECK_CALL(XLearnSetBool(&r_exptr,
                             CHAR(Rf_asChar(key)),
                             Rf_asLogical(value)));
    R_API_END();
}


