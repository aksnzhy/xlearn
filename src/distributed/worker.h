/*
 * worker.h
 * Copyright (C) 2017 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef WORKER_H
#define WORKER_H

#include "ps/ps.h"

namespace xlearn{

class XLearnWorker {
 public:
  XLearnWorker() {
    kv_w_ = new ps::KVWorker<float>(0);
    kv_v_ = new ps::KVWorker<float>(1);
  }
  ~XLearnWorker() {}

 private:
  ps::KVWorker<float>* kv_w_;
  ps::KVWorker<float>* kv_v_;
};

}

#endif /* !WORKER_H */
