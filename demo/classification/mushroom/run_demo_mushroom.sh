# Training task:
#  -s 0   (use logistic regression for classification)
#  -x acc (use accuracy metric)
# The model will be stored in agaricus_train.txt.model
../../xlearn_train ./agaricus_train.txt -s 0 -v ./agaricus_test.txt -x acc
# Prediction task:
# The output result will be stored in agaricus_test.txt.out
../../xlearn_predict ./agaricus_test.txt ./agaricus_train.txt.model 