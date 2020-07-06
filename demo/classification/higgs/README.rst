Higgs classification
---------------------------

You can find the full data from this here (`Link`__)

The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes. There is an interest in using deep learning methods to obviate the need for physicists to manually develop such features. Benchmark results using Bayesian Decision Trees from a standard physics package and 5-layer neural networks are presented in the original paper. The last 500,000 examples are used as a test set.

Popper
*****
There is a performace validation test that you can find in the popper folder that compares the liblinear and xLearn libraries with a workflow that automatically downloads the data set, runs the benchmark and shows the results on a chart.

.. __: https://archive.ics.uci.edu/ml/datasets/HIGGS
