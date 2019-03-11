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

REM # Training task:
REM #  -s 0   (use logistic regression for classification)
REM #  -x acc (use accuracy metric)
REM # The model will be stored in agaricus_train.txt.model
..\..\Release\xlearn_train.exe ./agaricus_train.txt -s 0 -v ./agaricus_test.txt -x acc
REM # Prediction task:
REM # The output result will be stored in agaricus_test.txt.out
..\..\Release\xlearn_predict.exe ./agaricus_test.txt ./agaricus_train.txt.model 