#!/bin/sh
set -ex

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark

echo "running benchmark"

python3 run_higgs_xlearn.py 

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="xlearn"

echo "RESULT,$result_name,$result,$USER,$start_fmt"

