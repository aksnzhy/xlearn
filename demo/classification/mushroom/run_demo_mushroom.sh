# Training task
../../xlearn_train ./agaricus_train.txt -s 0 -v ./agaricus_test.txt -x acc
# Prediction task
../../xlearn_predict ./agaricus_test.txt ./agaricus_train.txt.model 