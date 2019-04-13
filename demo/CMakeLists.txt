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
if(WIN32)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/run_demo_ctr.bat" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/criteo_ctr)
else()
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/run_demo_ctr.sh" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/criteo_ctr)
endif()
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/run_demo_ctr.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/criteo_ctr)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/criteo_ctr/run_demo_ctr.py" 
DESTINATION ${PROJECT_BINARY_DIR}/)

# Higgs
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/higgs/higgs-test.csv" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/higgs)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/higgs/higgs-train.csv" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/higgs)
if(WIN32)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/higgs/run_higgs.bat" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/higgs)
else()
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/higgs/run_higgs.sh" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/higgs)
endif()
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/higgs/run_higgs.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/higgs)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/higgs/run_higgs_pandas.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/higgs)

# Mushroom
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/mushroom/agaricus_test.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/mushroom)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/mushroom/agaricus_train.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/mushroom)
if(WIN32)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/mushroom/run_demo_mushroom.bat" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/mushroom)
else()
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/mushroom/run_demo_mushroom.sh" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/mushroom)
endif()
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/mushroom/run_demo_mushroom.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/mushroom)

# Titanic
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/titanic/titanic_train.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/titanic)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/titanic/titanic_test.txt" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/titanic)
if(WIN32)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/titanic/run_demo_titanic.bat" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/titanic)
else()
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/titanic/run_demo_titanic.sh" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/titanic)
endif()
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/titanic/run_demo_titanic.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/titanic)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/titanic/run_titanic_no_cv.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/titanic)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/classification/titanic/run_titanic_no_cv_pandas.py" 
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
if(WIN32)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/regression/house_price/run_demo_house_price.bat" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/house_price)
else()
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/regression/house_price/run_demo_house_price.sh" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/house_price)
endif()
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/regression/house_price/run_demo_house_price.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/house_price)
if(WIN32)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/regression/house_price/run_house_no_cv.bat" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/house_price)
else()
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/regression/house_price/run_house_no_cv.sh" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/house_price)
endif()
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/regression/house_price/run_house_no_cv.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/house_price)
FILE(COPY "${CMAKE_CURRENT_SOURCE_DIR}/regression/house_price/run_house_no_cv_pandas.py" 
DESTINATION ${PROJECT_BINARY_DIR}/demo/house_price)
