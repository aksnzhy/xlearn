#! /bin/sh
#
# run_train.sh
# Copyright (C) 2017 root <root@izt4ngd9bowt8sik34geskz>
#
# Distributed under terms of the MIT license.
#

echo "=====================================LR================================="
./xlearn_build/xlearn_train ~/xiaoshu/xlearn/demo/classification/criteo_ctr/small_train.txt -v ~/xiaoshu/xlearn/demo/classification/criteo_ctr/small_test.txt -s 0 -x auc --dis-es
echo "=====================================FM================================="
./xlearn_build/xlearn_train ~/xiaoshu/xlearn/demo/classification/criteo_ctr/small_train.txt -v ~/xiaoshu/xlearn/demo/classification/criteo_ctr/small_test.txt -s 1 -x auc --dis-es
echo "=====================================FFM==============================="
./xlearn_build/xlearn_train ~/xiaoshu/xlearn/demo/classification/criteo_ctr/small_train.txt -v ~/xiaoshu/xlearn/demo/classification/criteo_ctr/small_test.txt -s 2 -x auc --dis-es
#./xlearn_train ~/xiaoshu/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/xiaoshu/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 2 -x auc
