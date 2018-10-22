rm -r xlearn
cp -r R-package xlearn
cp -r src xlearn/src/src
R CMD build xlearn
R CMD INSTALL xlearn_0.3.4.tar.gz
