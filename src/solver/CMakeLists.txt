# Set output library.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
if (APPLE)
    set(CMAKE_MACOSX_RPATH 1)
endif ()

# Build static library
set(STA_DEPS reader loss score data base)
add_library(solver SHARED checker.cc trainer.cc inference.cc solver.cc coregistration.cc)
if (NOT WIN32)
    target_link_libraries(solver ${STA_DEPS})
else (WIN32)
    target_link_libraries(solver ${STA_DEPS} Ws2_32)
endif ()

# Build xlearn exe
if (NOT WIN32)
    set(LIBS solver reader loss score data base pthread)
    include_directories($ENV{JAVA_HOME}/include)
    if (APPLE)
        # Please use command below to find `jni_md.h`
        # find / -name jni_md.h 2> /dev/null
        include_directories($ENV{JAVA_HOME}/include/darwin)
    else ()
        include_directories($ENV{JAVA_HOME}/include/linux)
    endif ()
else (WIN32)
    set(LIBS solver reader loss score data base Ws2_32)
endif ()

add_executable(xlearn_train train_main.cc)
target_link_libraries(xlearn_train ${LIBS})

add_executable(xlearn_predict predict_main.cc)
target_link_libraries(xlearn_predict ${LIBS})

# Install library and header files
install(TARGETS solver DESTINATION lib/solver)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/solver)
