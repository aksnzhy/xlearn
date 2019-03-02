# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Build static library
set(STA_DEPS reader loss score data base)
add_library(solver STATIC checker.cc trainer.cc inference.cc solver.cc)
if(NOT WIN32)
target_link_libraries(solver ${STA_DEPS})
else(WIN32)
target_link_libraries(solver ${STA_DEPS} Ws2_32)
endif()

# Build xlearn exe
if(NOT WIN32)
set(LIBS solver reader loss score data base pthread)
else(WIN32)
set(LIBS solver reader loss score data base Ws2_32)
endif()

add_executable(xlearn_train train_main.cc)
target_link_libraries(xlearn_train ${LIBS})

add_executable(xlearn_predict predict_main.cc)
target_link_libraries(xlearn_predict ${LIBS})

# Install library and header files
install(TARGETS solver DESTINATION lib/solver)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/solver)
