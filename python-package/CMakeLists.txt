# Copy Python file
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/xlearn/base.py" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package/xlearn)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/xlearn/libpath.py" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package/xlearn)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/xlearn/xlearn.py" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package/xlearn)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/xlearn/_sklearn.py"
DESTINATION ${PROJECT_BINARY_DIR}/python-package/xlearn)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/xlearn/VERSION" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package/xlearn)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/xlearn/__init__.py" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package/xlearn)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/xlearn/data.py" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package/xlearn)

FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/MANIFEST.in" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package)
if(NOT WIN32)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/install-python.sh" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package)
else(WIN32)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/install-python.bat" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package)
endif()

FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/test/test_python.py" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package/test)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/test/test_data_conversion.py" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package/test)
