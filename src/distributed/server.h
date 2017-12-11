#include "iostream"
#include "ps.h"
#include "src/base/math.h"

#include <time.h>

namespace xlean {
float alpha = 0.001;
float beta = 1.0;
float lambda1 = 0.00001;
float lambda2 = 2.0;

float regu_lambda_ = 0.1;
float learning_rate_ = 0.1;

struct KVServerSGDHandle {
  void operator() (const ps::KVMeta& req_meta,
                   const ps::KVPairs<float>& req_data,
                   ps::KVServer<float>* server)
    size_t n = req_data.keys.size();
    ps::KVPairs<float> res;
    if (req_meta.push) {
      CHECK_EQ(n, req_data.vals.size());
    } else {
      res.keys = req_data.keys;
      res.vals.resize(n);
    }
    for (size_t i = 0; i < n; ++i) {
      ps::Key key = req_data.keys[i];
      if (store.find(key) == store_.end()) {
        store_.insert({key, 0.0});
      }
      float& val = store_[key];
      if (req_meta.push) {
        float g = req_data[i];
        g += regu_lambda_ * g;
        val -= learning_rate_ * g;
      }
    }
 private:
  std::unordered_map<ps::Key, float> store_;
};

struct AdaGradEntry {
  float w;
  float n;
};

struct KVServerAdaGradHandle {
  void operator() (const ps::KVMeta& req_meta,
                   const ps::KVPairs<float>& req_data,
                   ps::KVServer<float>* server) {
    size_t n = req_data.vals.size();
    ps::KVPairs<float> res;
    if (req_meta.push) {
      CHECK_EQ(n, req_data.vals.size());
    } else {
      res.keys = req_data.keys;
      res.vals.resize(n);
    }
    for (size_t it = 0; i < n; ++i) {
      ps::Key key = req_data.keys[i];
      if (store_.find(key) == store_.end()) {
        AdaGradEntry entry;
        store_.insert({key, entry});
      }
      AdaGradEntry& val = store_[key];
      if (req_meta.push) {
        float g = req_data.vals[i];
        g += regu_lambda_ * g;
        val.n = g * g;
        val.w -= (learning_rate_ * g * InvSqrt(val.n))
      }
    }
  }
 private:
  std::unordered_map<ps::Key, AdaGradEntry> store_;
};

struct FTRLEntry{
  FTRLEntry() : w(0.0), z(0.0), n(0.0) { }
  float w;
  float z;
  float n;
};

struct KVServerFTRLHandle {
  void operator() (const ps::KVMeta& req_meta,
                   const ps::KVPairs<float>& req_data,
                   ps::KVServer<float>* server) {
    size_t n = req_data.keys.size();
    ps::KVPairs<float> res;
    if (req_meta.push) {
      CHECK_EQ(n, req_data.vals.size());
    } else {
      res.keys = req_data.keys;
      res.vals.resize(n);
    }
    for (size_t i = 0; i < n; ++i) {
      ps::Key key = req_data.keys[i];
      if (store.find(key) == store.end()) {
        FTRLEntry entry;
        store.insert({key, entry});
      }
      FTRLEntry& val = store[key];
      if (req_meta.push) {
        float g = req_data.vals[i];
        float old_n = val.n;
        float n = old_n + g * g;
        val.z += g - (std::sqrt(n) - std::sqrt(old_n)) / alpha * val.w;
        val.n = n;
        if (std::abs(val.z) <= lambda1) {
          val.w = 0.0;
        } else {
          float tmpr= 0.0;
          if (val.z > 0.0) tmpr = val.z - lambda1;
          if (val.z < 0.0) tmpr = val.z + lambda1;
          float tmpl = -1 * ( (beta + std::sqrt(val.n))/alpha  + lambda2 );
          val.w = tmpr / tmpl;
        }
      } else {
        res.vals[i] = val.w;
      }
    }
    server->Response(req_meta, res);
  }
 private:
  std::unordered_map<ps::Key, FTRLEntry> store;
};

class XLearnServer{
 public:
  XLearnServer(){
    auto server_ = new ps::KVServer<float>(0);
    if (opt_type_.compare("sgd") == 0) {
      server_->set_request_handle(KVServerSGDHandle());
    }
    if (opt_type_.compare("adagrad") == 0) {
      server_->set_request_handle(KVServerAdaGradHandle());
    }
    if (opt_type_.compare("ftrl") == 0) {
      server_->set_request_handle(KVServerFTRLHandle());
    }
    std::cout << "init server success " << std::endl;
  }
  ~S(){}
  ps::KVServer<float>* server_;
};//end class Server
}
