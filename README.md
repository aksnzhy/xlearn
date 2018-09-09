<img src="https://github.com/aksnzhy/xLearn/raw/master/img/xlearn_logo.png" width = "400"/>

[![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)](./LICENCE)
![Project Status](https://img.shields.io/badge/version-0.3.1-green.svg)
[![Travis](https://img.shields.io/travis/rust-lang/rust.svg)]()


## What is xLearn?

xLearn is a ***high performance***, ***easy-to-use***, and ***scalable*** machine learning package, 
which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine
learning problems on large-scale sparse data, which is very common in Internet services such as online advertisement 
and recommender systems in recent years.  If you are the user  of liblinear, libfm, or libffm, now xLearn is your 
another better choice.

[Get Started! (English)](http://xlearn-doc.readthedocs.io/en/latest/index.html)

[Get Started! (中文)](http://xlearn-doc-cn.readthedocs.io/en/latest/index.html)

### Performance

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/speed.png" width = "800"/>

xLearn is developed with high-performance C++ code with careful design and optimizations. Our system is designed to maximize CPU and memory utilization, provide cache-aware computation, and support lock-free learning. By combining these insights, xLearn is 5x-13x faster compared to similar systems.

### Ease-of-use

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/code.png" width = "600"/>

xLearn does not rely on any troublesome third-party library, and hence users can just clone the code and compile it by using cmake. Also, xLearn supports very simple Python and R API for data scientists, and it also offers many useful features that have been widely used in machine learning and data mining competitions, such as cross-validation, early-stop, etc.

### Scalability

<img src="https://github.com/aksnzhy/xLearn/raw/master/img/scalability.png" width = "650"/>

xLearn can be used for solving large-scale machine learning problems. First, xLearn supports out-of-core training, which can handle very large data (TB) by just leveraging the disk of a PC. In addition, xLearn supports distributed training, which scales beyond billions of example across many machines by using the parameter server framework.


## News

 - 08/09/2018 xLearn uses the new logo:

 <img src="https://github.com/aksnzhy/xLearn/raw/master/img/xlearn_logo.png" width = "200"/>

 - 07/09/2018 The [Chinese document](http://xlearn-doc-cn.readthedocs.io/en/latest/index.html) is available now!

 - 08/03/2018 xLearn 0.3.0 version release. Main update:

    * Fix bugs in previous version
    * Solved the memory leak problem for on-disk learning
    * Support TXT model checkpoint
    * Support Scikit-Learn API

 - 18/12/2017 xLearn 0.2.0 version release. Main update:

    * Fix bugs in previous version
    * Support pip installation
    * New Documents
    * Faster FTRL algorithm

 - 11/24/2017 The first version (0.1.0) of xLearn release !
