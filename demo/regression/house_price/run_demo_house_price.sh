# Train model:
#  -s 4      (use factorization machine for regression)
#  -x rmse   (use RMSE metric)
#  -r 0.2    (set learning rate)
#  -b 0.002  (set regular lambda)
#  --cv      (use cross-validation)
../../xlearn_train ./house_price_train.txt -s 4 -x rmse -r 0.2 -b 0.002 --cv