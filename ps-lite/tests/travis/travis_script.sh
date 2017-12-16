#!/bin/bash
# main script of travis

if [ ${TASK} == "lint" ]; then
    make lint || exit -1
fi

if [ ${TASK} == "build" ]; then
    make DEPS_PATH=${CACHE_PREFIX} CXX=${CXX} || exit -1
fi

if [ ${TASK} == "test" ]; then
    make test DEPS_PATH=${CACHE_PREFIX} CXX=${CXX} || exit -1
    cd tests
    find test_* -type f -executable -exec ./repeat.sh 4 ./local.sh 2 2 ./{} \;
fi
