# Build library base
add_library(base_static STATIC stringprintf.cc logging.cc split_string.cc 
levenshtein_distance.cc timer.cc)

add_library(base SHARED stringprintf.cc logging.cc split_string.cc 
levenshtein_distance.cc timer.cc)

# Set properties
set_target_properties(base_static PROPERTIES OUTPUT_NAME "base")
set_target_properties(base PROPERTIES OUTPUT_NAME "base")

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
