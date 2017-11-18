# Build static library
set(STA_DEPS solver reader loss score data base)
add_library(c_api STATIC c_api.cc c_api_error.cc)
target_link_libraries(c_api ${STA_DEPS})

# Build shared library
set(SHA_DEPS solver_shared reader_shared loss_shared 
score_shared data_shared base_shared)
add_library(c_api_shared SHARED c_api.cc c_api_error.cc)
target_link_libraries(c_api_shared ${SHA_DEPS})

# Set properties
set_target_properties(c_api_shared PROPERTIES OUTPUT_NAME "c_api")
set_target_properties(c_api_shared PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(c_api PROPERTIES CLEAN_DIRECT_OUTPUT 1)

# Build unittests.
set(LIBS xlearn solver reader loss score data base pthread gtest)


# Install library and header files
install(TARGETS c_api DESTINATION lib/c_api)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/c_api)