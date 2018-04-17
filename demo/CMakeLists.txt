# Criteo CTR
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/small_test.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/criteo_ctr)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/small_test.txt" 
DESTINATION ${PROJECT_BINARY_DIR})
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/small_test.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package/test)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/small_train.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/criteo_ctr)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/small_train.txt" 
DESTINATION ${PROJECT_BINARY_DIR})
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/small_train.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/python-package/test)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/run_demo_ctr.sh" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/criteo_ctr)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/run_demo_ctr.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/criteo_ctr)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/run_demo_ctr.py" 
DESTINATION ${PROJECT_BINARY_DIR}/)

# Mushroom
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/mushroom/agaricus_test.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/mushroom)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/mushroom/agaricus_train.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/mushroom)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/mushroom/run_demo_mushroom.sh" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/mushroom)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/mushroom/run_demo_mushroom.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/mushroom)

# Titanic
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/titanic/titanic_train.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/titanic)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/titanic/titanic_test.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/titanic)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/titanic/run_demo_titanic.sh" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/titanic)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/titanic/run_demo_titanic.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/titanic)

# Scikit-Learn
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/scikit_learn_demo/example_FFM_criteo.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/scikit_learn_demo)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/scikit_learn_demo/example_FM_wine.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/scikit_learn_demo)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/scikit_learn_demo/example_LR_iris.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/scikit_learn_demo)

# House price
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/regression/house_price/house_price_train.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/house_price)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/regression/house_price/house_price_test.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/house_price)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/regression/house_price/run_demo_house_price.sh" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/house_price)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/regression/house_price/run_demo_house_price.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/house_price)