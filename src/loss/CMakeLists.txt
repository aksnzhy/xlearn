# Build library loss
add_library(loss loss.cc squared_loss.cc cross_entropy_loss.cc metric.cc)

# Build uinttests
set(LIBS loss gtest base data score)

add_executable(loss_test loss_test.cc)
target_link_libraries(loss_test gtest_main ${LIBS})

add_executable(squared_loss_test squared_loss_test.cc)
target_link_libraries(squared_loss_test gtest_main ${LIBS})

add_executable(cross_entropy_loss_test cross_entropy_loss_test.cc)
target_link_libraries(cross_entropy_loss_test gtest_main ${LIBS})

add_executable(metric_test metric_test.cc)
target_link_libraries(metric_test gtest_main ${LIBS})

# Install library and header files
install(TARGETS loss DESTINATION lib/loss)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/loss)
