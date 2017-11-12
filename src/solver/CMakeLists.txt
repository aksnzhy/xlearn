# Build static library
add_library(solver STATIC checker.cc trainer.cc inference.cc solver.cc)

# Build xlearn exe
set(LIBS solver reader loss score data base pthread)

add_executable(xlearn_train train_main.cc)
target_link_libraries(xlearn_train ${LIBS})

add_executable(xlearn_predict predict_main.cc)
target_link_libraries(xlearn_predict ${LIBS})

# Install library and header files
install(TARGETS solver DESTINATION lib/solver)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/solver)
