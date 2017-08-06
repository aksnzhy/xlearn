# Build library loss
add_library(loss loss.cc)

# Build uinttests
set(LIBS loss gtest base data score)

add_executable(loss_test loss_test.cc)
target_link_libraries(loss_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS loss DESTINATION lib/loss)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/loss)
