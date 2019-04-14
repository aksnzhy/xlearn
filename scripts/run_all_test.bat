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

REM # This script runs all of the unit test for C++
.\base\Release\file_util_test.exe
.\base\Release\levenshtein_distance_test.exe
.\base\Release\thread_pool_test.exe
.\c_api\Release\c_api_test.exe
.\data\Release\data_structure_test.exe
.\data\Release\model_parameters_test.exe
.\loss\Release\cross_entropy_loss_test.exe
.\loss\Release\loss_test.exe
.\loss\Release\metric_test.exe
.\loss\Release\squared_loss_test.exe
.\reader\Release\file_splitor_test.exe
.\reader\Release\parser_test.exe
.\reader\Release\reader_test.exe
.\score\Release\ffm_score_test.exe
.\score\Release\fm_score_test.exe
.\score\Release\linear_score_test.exe
.\score\Release\score_function_test.exe