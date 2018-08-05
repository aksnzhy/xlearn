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

# Train model:
#  -s 4      (use factorization machine for regression)
#  -x rmse   (use RMSE metric)
#  -r 0.2    (set learning rate)
#  -b 0.002  (set regular lambda)
#  --cv      (use cross-validation)
../../xlearn_train ./house_price_train.txt -s 4 -x rmse -r 0.2 -b 0.002 --cv