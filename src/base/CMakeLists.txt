# Build library base
add_library(base logging.cc split_string.cc stringprintf.cc)

# Install library and header files
install(TARGETS base DESTINATION lib/base)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/base)
