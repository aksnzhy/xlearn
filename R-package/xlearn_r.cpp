/*
 * xlearn.cpp
 * Copyright (C) 2017 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */
#include <R.h>
#include <Rmath.h>

//#include "src/base/common.h"
//#include "src/data/hyper_parameters.h"
//#include "src/solver/solver.h"
#include "/Users/xiaoshuwang/documents/oneflow/opensource/xlearn/src/c_api/c_api.h"

extern "C" int create_r(const char* model_type, XLearnHandle *out) {
  int res = XLearnCreate(model_type, out);
  return res;
}

extern "C" void show_r() {
  xl = new XLearn;
  int res = create_r("fm", xl);
  std::cout << "show_r : " << xl->GetHyperParam().score_func.c_str() << std::endl;
}

