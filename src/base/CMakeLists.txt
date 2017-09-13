# Build library base
add_library(base logging.cc split_string.cc stringprintf.cc levenshtein_distance.cc)

# Build unittests.
set(LIBS base gtest)

add_executable(levenshtein_distance_test levenshtein_distance_test.cc)
target_link_libraries(levenshtein_distance_test gtest_main ${LIBS})

add_executable(file_util_test file_util_test.cc)
target_link_libraries(file_util_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS base DESTINATION lib/base)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/base)
