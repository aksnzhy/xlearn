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
    w.resize(k, 0.0);
  }
  std::vector<float> w;
}

struct KVServerSGDHandle {
  void operator() (const ps::KVMeta& req_meta,
                   const ps::KVPairs<float>& req_data,
                   ps::KVServer<float>* server)
    int k = 0;
    if (req_data.lens() == 0) {
      k = req_data.vals.size() / req_data.keys.size();
    }
    size_t keys_size = req_data.keys.size();
    ps::KVPairs<float> res;
    if (req_meta.push) {
      CHECK_EQ(keys_size * k, req_data.vals.size());
    } else {
      res.keys = req_data.keys;
      res.vals.resize(keys_size);
    }
    for (size_t i = 0; i < keys_size; ++i) {
      ps::Key key = req_data.keys[i];
      SGDEntry& val = store_[key];
      if (req_meta.push) {
        for (int j = 0; j < val.w.size(); ++j) {
          float gradient = req_data.vals[i * k + j];
          gradient += regu_lambda_ * gradient;
          val.w[j] -= learning_rate_ * gradient;
        }
      } else {
        for (int j = 0; j < val.w.size(); ++j) {
          res.vals[i * k + j] = val.w[j];
        }
      }
    }
 private:
  std::unordered_map<ps::Key, SGDEntry> store_;
};

struct AdaGradEntry {
  AdaGradEntry(size_t k) {
    w.resize(k, 0.0);
    n.resize(k, 0.0);
  }
  std::vector<float> w;
  std::vector<float> n;
};

struct KVServerAdaGradHandle {
  void operator() (const ps::KVMeta& req_meta,
                   const ps::KVPairs<float>& req_data,
                   ps::KVServer<float>* server) {
    size_t k = 0;
    if (req_data.lens.size() == 0) {
      k = req_data.vals.size() / req_data.keys.size();
    }
    size_t keys_size = req_data.keys.size();
    ps::KVPairs<float> res;
    if (req_meta.push) {
      CHECK_EQ(keys_size * k, req_data.vals.size());
    } else {
      res.keys = req_data.keys;
      res.vals.resize(n);
    }
    for (size_t it = 0; i < n; ++i) {
      ps::Key key = req_data.keys[i];
      AdaGradEntry& val = store_[key];
      if (req_meta.push) {
        for (int j = 0; j < val.w.size(); ++j) {
          float g = req_data.vals[i * k + j];
          g += regu_lambda_ * g;
          val.n[j] = g * g;
          val.w[j] -= (learning_rate_ * g * InvSqrt(val.n[j]))
        }
      } else {
        for (int j = 0; j < val.w.size(); ++j) {
          res[i * k + j] = val.w[j];
        }
      }
    }
  }
 private:
  std::unordered_map<ps::Key, AdaGradEntry> store_;
};

struct FTRLEntry{
  FTRLEntry(int k) {
    w.resize(k, 0.0);
    n.resize(k, 0.0);
    z.resize(k, 0.0);
  }
  std::vector<float> w;
  std::vector<float> z;
  std::vector<float> n;
};

struct KVServerFTRLHandle {
  void operator() (const ps::KVMeta& req_meta,
                   const ps::KVPairs<float>& req_data,
                   ps::KVServer<float>* server) {
    int k = 0; 
    if (req_data.lens.size() == 0) {
      k = req_data.vals.size() / req_data.keys.size();
    }
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
      FTRLEntry& val = store[key];
      for (int j = 0; j < val.w.size(); ++j) {
        if (req_meta.push) {
          float g = req_data.vals[i * k + j];
          float old_n = val.n[j];
          float n = old_n + g * g;
          val.z[j] += g - (std::sqrt(n) - std::sqrt(old_n)) / alpha * val.w[j];
          val.n[j] = n;
          if (std::abs(val.z[j]) <= lambda1) {
            val.w[j] = 0.0;
          } else {
            float tmpr= 0.0;
            if (val.z[j] > 0.0) tmpr = val.z[j] - lambda1;
            if (val.z[j] < 0.0) tmpr = val.z[j] + lambda1;
            float tmpl = -1 * ( (beta + std::sqrt(val.n[j]))/alpha  + lambda2 );
            val.w[j] = tmpr / tmpl;
          }
        } else {
          res.vals[i * k + j] = val.w[j];
        }
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
