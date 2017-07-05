# Build library loss
add_library(loss score_function.cc linear_score.cc)

# Build uinttests
set(LIBS loss gtest base)

add_executable(linear_score_test linear_score_test.cc)
target_link_libraries(linear_score_test gtest_main ${LIBS})
