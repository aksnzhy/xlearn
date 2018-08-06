//------------------------------------------------------------------------------
// Copyright (c) 2018 by contributors. All Rights Reserved.
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
This file is the implementation of the Metric class.
*/

#include "src/loss/metric.h"

namespace xLearn {

CLASS_REGISTER_IMPLEMENT_REGISTRY(xLearn_metric_registry, Metric);
REGISTER_METRIC("acc", AccMetric);
REGISTER_METRIC("prec", PrecMetric);
REGISTER_METRIC("recall", RecallMetric);
REGISTER_METRIC("f1", F1Metric);
REGISTER_METRIC("mae", MAEMetric);
REGISTER_METRIC("mape", MAPEMetric);
REGISTER_METRIC("rmsd", RMSDMetric);
REGISTER_METRIC("auc", AUCMetric);

}  // namespace xLearn
