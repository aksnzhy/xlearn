# Build library updater
add_library(updater updater.cc adagrad_updater.cc momentum_updater.cc adadelta_updater.cc rmsprop_updater.cc)

# Build unittests.
set(LIBS updater gtest base data)

add_executable(updater_test updater_test.cc)
target_link_libraries(updater_test gtest_main ${LIBS})

add_executable(momentum_updater_test momentum_updater_test.cc)
target_link_libraries(momentum_updater_test gtest_main ${LIBS})

add_executable(adagrad_updater_test adagrad_updater_test.cc)
target_link_libraries(adagrad_updater_test gtest_main ${LIBS})

add_executable(rmsprop_updater_test rmsprop_updater_test.cc)
target_link_libraries(rmsprop_updater_test gtest_main ${LIBS})

add_executable(adadelta_updater_test adadelta_updater_test.cc)
target_link_libraries(adadelta_updater_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS updater DESTINATION lib/update)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/updater)
