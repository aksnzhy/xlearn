# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/score)

# Build static library
set(STA_DEPS data base)
add_library(score STATIC score_function.cc 
linear_score.cc fm_score.cc ffm_score.cc)
target_link_libraries(score ${STA_DEPS})

# Build uinttests
set(LIBS score data base gtest)

add_executable(score_function_test score_function_test.cc)
target_link_libraries(score_function_test gtest_main ${LIBS})

add_executable(linear_score_test linear_score_test.cc)
target_link_libraries(linear_score_test gtest_main ${LIBS})

add_executable(fm_score_test fm_score_test.cc)
target_link_libraries(fm_score_test gtest_main ${LIBS})

add_executable(ffm_score_test ffm_score_test.cc)
target_link_libraries(ffm_score_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS score DESTINATION lib/score)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/score)
