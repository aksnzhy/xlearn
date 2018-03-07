# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/c_api)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Build static library
set(STA_DEPS solver reader loss score data base)
add_library(xlearn_api STATIC c_api.cc c_api_error.cc)
target_link_libraries(xlearn_api ${STA_DEPS})

# Build shared library
add_library(xlearn_api_shared SHARED c_api.cc c_api_error.cc 
../base/logging.cc ../base/stringprintf.cc ../base/split_string.cc 
../base/levenshtein_distance.cc ../base/timer.cc 
../data/model_parameters.cc 
../loss/loss.cc ../loss/squared_loss.cc ../loss/cross_entropy_loss.cc 
../loss/metric.cc 
../reader/parser.cc ../reader/file_splitor.cc ../reader/reader.cc 
../score/score_function.cc ../score/linear_score.cc ../score/fm_score.cc 
../score/ffm_score.cc 
../solver/checker.cc ../solver/trainer.cc 
../solver/inference.cc ../solver/solver.cc)

# Set properties
set_target_properties(xlearn_api_shared PROPERTIES OUTPUT_NAME "xlearn_api")
set_target_properties(xlearn_api_shared PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(xlearn_api PROPERTIES CLEAN_DIRECT_OUTPUT 1)

# Build unittests.
set(LIBS xlearn_api solver reader loss score data base pthread gtest)

add_executable(c_api_test c_api_test.cc)
target_link_libraries(c_api_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS xlearn_api DESTINATION lib/c_api)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/c_api)