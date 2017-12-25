# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/reader)

# Build static library
set(STA_DEPS data base)
add_library(reader STATIC parser.cc file_splitor.cc reader.cc)
target_link_libraries(reader ${STA_DEPS})

# Build uinttests.
set(LIBS reader data base pthread gtest)

add_executable(parser_test parser_test.cc)
target_link_libraries(parser_test gtest_main ${LIBS})

add_executable(reader_test reader_test.cc)
target_link_libraries(reader_test gtest_main ${LIBS})

add_executable(file_splitor_test file_splitor_test.cc)
target_link_libraries(file_splitor_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS reader DESTINATION lib/reader)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/reader)
