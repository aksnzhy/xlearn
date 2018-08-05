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

# Training task:
#  -s 0   (use logistic regression for classification)
#  -x acc (use accuracy metric)
# The model will be stored in agaricus_train.txt.model
../../xlearn_train ./agaricus_train.txt -s 0 -v ./agaricus_test.txt -x acc
# Prediction task:
# The output result will be stored in agaricus_test.txt.out
../../xlearn_predict ./agaricus_test.txt ./agaricus_train.txt.model 