# Training task:
#  -s : 2    (use ffm for classification)
#  -x : acc  (use accuracy metric)
# The model will be stored in small_train.txt.model
../../xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -x acc
# Prediction task:
# The output result will be stored in small_test.txt.out
../../xlearn_predict ./small_test.txt ./small_train.txt.model