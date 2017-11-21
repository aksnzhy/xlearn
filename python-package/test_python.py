# This file test the xlearn python package.
# We create a ffm model for binary classification problem.
# The dataset comes from the criteo CTR  .
import xlearn as xl

ffm_model = fm.create_ffm()
ffm_model.serTrain(..)
ffm_model.setValidate(..)