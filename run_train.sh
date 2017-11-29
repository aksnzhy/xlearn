#! /bin/sh
#
# run_train.sh
# Copyright (C) 2017 root <root@izt4ngd9bowt8sik34geskz>
#
# Distributed under terms of the MIT license.
#
echo "====================================SmallData LR==================================="
#/Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/demo/classification/criteo_ctr/small_train.txt -v /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/demo/classification/criteo_ctr/small_test.txt -s 0 -x auc --dis-es
echo "====================================SmallData FM==================================="
#/Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/demo/classification/criteo_ctr/small_train.txt -v /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/demo/classification/criteo_ctr/small_test.txt -s 1 -x auc --dis-es
echo "====================================SmallData FFM=================================="
#/Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/demo/classification/criteo_ctr/small_train.txt -v /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/demo/classification/criteo_ctr/small_test.txt -s 2 -x auc --dis-es

echo "=====================================BigDataLR================================="
/Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 0 -x auc --dis-es
echo "=====================================BigDataFM================================="
/Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 1 -x auc --dis-es
echo "=====================================BigDataFFM==============================="
#/Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 2 -x auc --dis-es

