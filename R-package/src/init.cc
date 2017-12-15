#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .Call calls */
extern SEXP XLearnCreate_R(SEXP);
extern SEXP XLearnFit_R(SEXP, SEXP);
extern SEXP XLearnPredict_R(SEXP, SEXP, SEXP);
extern SEXP XLearnSetFloat_R(SEXP, SEXP, SEXP);
extern SEXP XLearnSetInt_R(SEXP, SEXP, SEXP);
extern SEXP XLearnSetStr_R(SEXP, SEXP, SEXP);
extern SEXP XLearnSetTrain_R(SEXP, SEXP);
extern SEXP XLearnSetValidate_R(SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
  {"XLearnCreate_R",      (DL_FUNC) &XLearnCreate_R,      1},
  {"XLearnFit_R",         (DL_FUNC) &XLearnFit_R,         2},
  {"XLearnPredict_R",     (DL_FUNC) &XLearnPredict_R,     3},
  {"XLearnSetFloat_R",    (DL_FUNC) &XLearnSetFloat_R,    3},
  {"XLearnSetInt_R",      (DL_FUNC) &XLearnSetInt_R,      3},
  {"XLearnSetStr_R",      (DL_FUNC) &XLearnSetStr_R,      3},
  {"XLearnSetTrain_R",    (DL_FUNC) &XLearnSetTrain_R,    2},
  {"XLearnSetValidate_R", (DL_FUNC) &XLearnSetValidate_R, 2},
  {NULL, NULL, 0}
};

extern "C" void R_init_xlearn(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}