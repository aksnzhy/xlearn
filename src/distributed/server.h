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

struct SGDEntry{
  SGDEntry(size_t k) {
    w.resize(w, 0.0);
  }
  std::vector<float> w;
}

struct KVServerSGDHandle {
  void operator() (const ps::KVMeta& req_meta,
                   const ps::KVPairs<float>& req_data,
                   ps::KVServer<float>* server)
    int k = 0;
    if (req_data.lens() == 0) {
      k = req_data.keys.size() / req_data.vals.size();
    }
    size_t keys_size = req_data.keys.size();
    ps::KVPairs<float> res;
    if (req_meta.push) {
      CHECK_EQ(keys_size * k, req_data.vals.size());
    } else {
      res.keys = req_data.keys;
      SGDEntry entry(k);
      res.vals.resize(keys_size, entry);
    }
    for (size_t i = 0; i < keys_size; ++i) {
      ps::Key key = req_data.keys[i];
      SGDEntry& val = store_[key];
      if (req_meta.push) {
        for (int j = 0; j < val.w.size(); ++j)
        float gradient = req_data[i * k + j];
        gradient += regu_lambda_ * gradient;
        val.w[j] -= learning_rate_ * gradient;
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
