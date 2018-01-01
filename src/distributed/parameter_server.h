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

This file defines the KVStore class, which allows workers to get
and set the model parameters.
*/

#ifndef XLEARN_DISTRIBUTED_KVSTORE_H_
#define XLEARN_DISTRIBUTED_KVSTORE_H_

#include <vector>

#include "src/base/common.h"

namespace xLearn {

//------------------------------------------------------------------------------
// KVStore are used for distributed training and it allows workers to get
// and set the model parameters by using pull() and push() API.
//------------------------------------------------------------------------------
class KVStore {
 public:
   // Constructor and Destructor
   KVStore() { }
   ~KVStore() { }

   // Push a list of (feature id, value) into store
   virtual void Push(const std::vector<index_t>& feat_list,
   	                 const std::vector<real_t>& value_list);

   // Pull the values for a list of feature ids
   virtual void Pull(const std::vector<index_t>& feat_list,
   	                 const std::vector<real_t>* value_list);
};

}  // namespace xLearn

#endif  // XLEARN_DISTRIBUTED_KVSTORE_H_