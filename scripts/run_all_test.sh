# Copyright (c) 2018 by contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
# This script runs all of the unit test for C++
./base/file_util_test
./base/levenshtein_distance_test
./base/thread_pool_test
./c_api/c_api_test
./data/data_structure_test
./data/model_parameters_test
./loss/cross_entropy_loss_test
./loss/loss_test
./loss/metric_test
./loss/squared_loss_test
./reader/file_splitor_test
./reader/parser_test
./reader/reader_test
./score/ffm_score_test
./score/fm_score_test
./score/linear_score_test
./score/score_function_test