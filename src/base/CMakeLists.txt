# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/base)

# Build static library
add_library(base STATIC logging.cc stringprintf.cc split_string.cc 
levenshtein_distance.cc timer.cc)

# Build shared library
add_library(base_shared SHARED logging.cc stringprintf.cc split_string.cc 
levenshtein_distance.cc timer.cc)

# Set properties
set_target_properties(base_shared PROPERTIES OUTPUT_NAME "base")
set_target_properties(base_shared PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(base PROPERTIES CLEAN_DIRECT_OUTPUT 1)

# Build unittests.
set(LIBS base pthread gtest)

add_executable(levenshtein_distance_test levenshtein_distance_test.cc)
target_link_libraries(levenshtein_distance_test gtest_main ${LIBS})

add_executable(file_util_test file_util_test.cc)
target_link_libraries(file_util_test gtest_main ${LIBS})

add_executable(thread_pool_test thread_pool_test.cc)
target_link_libraries(thread_pool_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS base DESTINATION lib/base)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/base)
