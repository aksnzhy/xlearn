<img src="https://github.com/aksnzhy/xLearn/raw/master/img/xlearn_logo.png" width = "400"/>

[![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)](./LICENCE)
[![Project Status](https://img.shields.io/badge/version-0.4.4-green.svg)]()

## What is xLearn?

xLearn is a ***high performance***, ***easy-to-use***, and ***scalable*** machine learning package, including linear model (LR), factorization machines (FM), and field-aware factorization machines (FFM), which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine learning problems on large-scale sparse data. Many real world datasets deal with high dimensional sparse feature vectors like a recommendation system where the number of categories and users is on the order of millions. In that case, if you are the user of liblinear, libfm, and libffm, now xLearn is your another better choice.

[Get Started! (English)](http://xlearn-doc.readthedocs.io/en/latest/index.html)

[Get Started! (中文)](http://xlearn-doc-cn.readthedocs.io/en/latest/index.html)

### Performance

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/speed.png" width = "800"/>

xLearn is developed by high-performance C++ code with careful design and optimizations. Our system is designed to maximize CPU and memory utilization, provide cache-aware computation, and support lock-free learning. By combining these insights, xLearn is 5x-13x faster compared to similar systems.

### Ease-of-use

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/code.png" width = "600"/>

xLearn does not rely on any troublesome third-party library, and hence users can just clone the code and compile it by using cmake. Also, xLearn supports very simple Python and CLI interface for data scientists, and it also offers many useful features that have been widely used in machine learning and data mining competitions, such as cross-validation, early-stop, etc.

### Scalability

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/scalability.png" width = "650"/>

xLearn can be used for solving large-scale machine learning problems. First, xLearn supports out-of-core training, which can handle very large data (TB) by just leveraging the disk of a PC. In addition, xLearn supports distributed training, which scales beyond billions of example across many machines by using the Parameter Server framework.

## How to Contribute

xLearn has been developed and used by many active community members. Your help is very valuable to make it better for everyone.

 * Please contribute if you find any bug in xLearn.
 * Contribute to the tests to make it more reliable.
 * Contribute to the documents to make it clearer for everyone.
 * Contribute to the examples to share your experience with other users.
 * Open issue if you met problems during development.

 Note that, please post iusse and contribution in *English* so that everyone can get help from them.

### Contributors (rank randomly)

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/10520307.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/11278017.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/1289856.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/13925796.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/15322665.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/1842965.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/21072881.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/22660103.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/2387719.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/25626965.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/3086744.png" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/3928409.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/4606937.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/6054101.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/6161143.png" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/7145046.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/7608904.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/27916175.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/7608904.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/1443518.png" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/9783213.png" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/4609798.jpeg" width = "40"/><img src="https://github.com/aksnzhy/xLearn/raw/master/img/11628637.png" width = "40"/>

## For Enterprise Users and Call for Sponsors

If you are enterprise users and find xLearn is useful in your work, please let us know, and we are glad to add your company logo here. We also welcome you become a sponsor to make this project better.

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/tencent.png" width = "200"/>
<img src="https://github.com/aksnzhy/xLearn/raw/master/img/stategrid.jpg" width = "200"/>
<img src="https://github.com/aksnzhy/xLearn/raw/master/img/xiaodaka.png" width = "200"/>

## What's New

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
