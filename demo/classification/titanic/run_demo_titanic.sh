# Train model:
#  -s 1   (use factorizarion machone for classification)
#  -x acc (use accuracy metric)
#  --cv   (use cross-validation)
../../xlearn_train ./titanic_train.txt -s 1 -x acc --cv