@echo off
REM # Copyright (c) 2018 by contributors. All Rights Reserved.
REM #
REM # Licensed under the Apache License, Version 2.0 (the "License");
REM # you may not use this file except in compliance with the License.
REM # You may obtain a copy of the License at
REM #
REM #     http://www.apache.org/licenses/LICENSE-2.0
REM #
REM # Unless required by applicable law or agreed to in writing, software
REM # distributed under the License is distributed on an "AS IS" BASIS,
REM # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM # See the License for the specific language governing permissions and
REM # limitations under the License.

REM # Train model:
REM #  -s 4      (use factorization machine for regression)
REM #  -x rmse   (use RMSE metric)
REM #  -r 0.2    (set learning rate)
REM #  -b 0.002  (set regular lambda)
REM #  --cv      (use cross-validation)
..\..\Release\xlearn_train.exe ./house_price_train.txt -s 4 -x rmse -r 0.2 -b 0.002 --cv