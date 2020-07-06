from liblinearutil import *

# Read data in LIBSVM format
y, x = svm_read_problem('HIGGSlibsvm')
m = train(y[:8800000], x[:8800000], '-s 0 -c 4 -B 1')
p_label, p_acc, p_val = predict(y[8800000:], x[8800000:], m) 
