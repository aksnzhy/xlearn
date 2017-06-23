# Build library updater
add_library(updater updater.cc momentum_updater.cc nesterov_updater.cc)

# Build unittests.
set(LIBS updater gtest base data)

add_executable(updater_test updater_test.cc)
target_link_libraries(updater_test gtest_main ${LIBS})

add_executable(momentum_updater_test momentum_updater_test.cc)
target_link_libraries(momentum_updater_test gtest_main ${LIBS})

add_executable(nesterov_updater_test nesterov_updater_test.cc)
target_link_libraries(nesterov_updater_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS updater DESTINATION lib/update)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/updater)
