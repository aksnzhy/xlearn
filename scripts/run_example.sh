# Training task
./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -x acc
# Prediction task
./xlearn_predict ./small_test.txt ./small_train.txt.model --sigmoid
