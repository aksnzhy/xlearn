set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/distributed)

set(STA_DEPS data base pslite protobuf zmq)
add_library(distributed STATIC dist_score_function.cc dist_linear_score.cc dist_cross_entropy_loss.cc dist_checker.cc dist_loss.cc)
target_link_libraries(distributed ${STA_DEPS})

add_executable(dist_xlearn_train dist_train_main.cc dist_solver.cc dist_trainer.cc dist_inference.cc)
target_link_libraries(dist_xlearn_train solver distributed)
