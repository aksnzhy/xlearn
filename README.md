<img src="https://github.com/aksnzhy/xLearn/raw/master/img/xlearn_logo.png" width = "400"/>

[![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)](./LICENCE)
[![Project Status](https://img.shields.io/badge/version-0.4.4-green.svg)]()

## What is xLearn?

xLearn is a ***high performance***, ***easy-to-use***, and ***scalable*** machine learning package that contains linear model (LR), factorization machines (FM), and field-aware factorization machines (FFM), all of which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine learning problems on large-scale sparse data. Many real world datasets deal with high dimensional sparse feature vectors like a recommendation system where the number of categories and users is on the order of millions. In that case, if you are the user of liblinear, libfm, and libffm, now xLearn is your another better choice.

[Get Started! (English)](http://xlearn-doc.readthedocs.io/en/latest/index.html)

[Get Started! (中文)](http://xlearn-doc-cn.readthedocs.io/en/latest/index.html)

### Performance

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/speed.png" width = "800"/>

xLearn is developed by high-performance C++ code with careful design and optimizations. Our system is designed to maximize CPU and memory utilization, provide cache-aware computation, and support lock-free learning. By combining these insights, xLearn is 5x-13x faster compared to similar systems.

### Ease-of-use

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/code.png" width = "600"/>

xLearn does not rely on any third-party library and users can just clone the code and compile it by using cmake. Also, xLearn supports very simple Python and CLI interface for data scientists, and it also offers many useful features that have been widely used in machine learning and data mining competitions, such as cross-validation, early-stop, etc.

### Scalability

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/scalability.png" width = "650"/>

xLearn can be used for solving large-scale machine learning problems. xLearn supports out-of-core training, which can handle very large data (TB) by just leveraging the disk of a PC.

## How to Contribute

xLearn has been developed and used by many active community members. Your help is very valuable to make it better for everyone.

 * Please contribute if you find any bug in xLearn.
 * Contribute new features you want to see in xLearn.
 * Contribute to the tests to make it more reliable.
 * Contribute to the documents to make it clearer for everyone.
 * Contribute to the examples to share your experience with other users.
 * Open issue if you met problems during development.

Note that, please post iusse and contribution in *English* so that everyone can get help from them.

## What's New

 - 2019-10-13 [Andrew Kane](https://github.com/ankane) add [Ruby bindings](https://github.com/ankane/xlearn) for xLearn!

 - 2019-4-25 xLearn 0.4.4 version release. Main update:

    * Support Python DMatrix
    * Better Windows support
    * Fix bugs in previous version

 - 2019-3-25 xLearn 0.4.3 version release. Main update:
    * Fix bugs in previous version

 - 2019-3-12 xLearn 0.4.2 version release. Main update:
    * Release Windows version of xLearn

 - 2019-1-30 xLearn 0.4.1 version release. Main update:
    * More flexible data reader

 - 2018-11-22 xLearn 0.4.0 version release. Main update:

    * Fix bugs in previous version
    * Add online learning for xLearn

 - 2018-11-10 xLearn 0.3.8 version release. Main update:

    * Fix bugs in previous version.
    * Update early-stop mechanism.

 - 2018-11-08. xLearn gets 2000 star! Congs!

 - 2018-10-29 xLearn 0.3.7 version release. Main update:

    * Add incremental Reader, which can save 50% memory cost.

 - 2018-10-22 xLearn 0.3.5 version release. Main update:

    * Fix bugs in 0.3.4.

 - 2018-10-21 xLearn 0.3.4 version release. Main update:

    * Fix bugs in on-disk training.
    * Support new file format.

 - 2018-10-14 xLearn 0.3.3 version release. Main update:

    * Fix segmentation fault in prediction task.
    * Update early-stop meachnism.

 - 2018-09-21 xLearn 0.3.2 version release. Main update:

    * Fix bugs in previous version
    * New TXT format for model output

 - 2018-09-08 xLearn uses the new logo:

 <img src="https://github.com/aksnzhy/xLearn/raw/master/img/xlearn_logo.png" width = "300"/>

 - 2018-09-07 The [Chinese document](http://xlearn-doc-cn.readthedocs.io/en/latest/index.html) is available now!

 - 2018-03-08 xLearn 0.3.0 version release. Main update:

    * Fix bugs in previous version
    * Solved the memory leak problem for on-disk learning
    * Support TXT model checkpoint
    * Support Scikit-Learn API

 - 2017-12-18 xLearn 0.2.0 version release. Main update:

    * Fix bugs in previous version
    * Support pip installation
    * New Documents
    * Faster FTRL algorithm

 - 2017-11-24 The first version (0.1.0) of xLearn release !
