# Build library trainer
add_library(train trainer.cc)

# Build xlearn exe
set(LIBS train base data loss reader score updater)

add_executable(xlearn main.cc)
target_link_libraries(xlearn ${LIBS})

# Install library and header files
install(TARGETS train DESTINATION lib/train)
FILE(GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADER_FILES} DESTINATION include/train)
