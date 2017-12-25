# Copy all of the files to R-package
file(COPY "../src/base" DESTINATION "./src/")
file(COPY "../src/c_api" DESTINATION "./src/")
file(COPY "../src/data" DESTINATION "./src/")
file(COPY "../src/loss" DESTINATION "./src/")
file(COPY "../src/reader" DESTINATION "./src/")
file(COPY "../src/score" DESTINATION "./src/")
file(COPY "../src/solver" DESTINATION "./src/")
file(COPY "../R-package/man" DESTINATION "./")
file(COPY "../R-package/R" DESTINATION "./")
file(COPY "../R-package/DESCRIPTION" DESTINATION "./")
file(COPY "../R-package/NAMESPACE" DESTINATION "./")
file(COPY "../R-package/src/.Rhistory" DESTINATION "./src")
file(COPY "../R-package/src/init.cc" DESTINATION "./src")
file(COPY "../R-package/src/Makevars" DESTINATION "./src")
file(COPY "../R-package/src/xlearn_R.cc" DESTINATION "./src")
file(COPY "../R-package/src/xlearn_R.h" DESTINATION "./src")

# Build shared library
add_library(xlearn SHARED ./src/init.cc ./src/xlearn_R.cc
./src/c_api/c_api.cc ./src/c_api/c_api_error.cc 
./src/base/logging.cc ./src/base/stringprintf.cc ./src/base/split_string.cc
./src/base/levenshtein_distance.cc ./src/base/timer.cc
./src/data/model_parameters.cc ./src/loss/loss.cc 
./src/loss/squared_loss.cc ./src/loss/cross_entropy_loss.cc
./src/loss/metric.cc
./src/reader/parser.cc ./src/reader/file_splitor.cc ./src/reader/reader.cc
./src/score/score_function.cc ./src/score/linear_score.cc ./src/score/fm_score.cc
./src/score/ffm_score.cc
./src/solver/checker.cc ./src/solver/trainer.cc
./src/solver/inference.cc ./src/solver/solver.cc)

# Set properties
set_target_properties(xlearn PROPERTIES PREFIX "")