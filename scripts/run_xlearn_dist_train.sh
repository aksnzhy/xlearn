#! /bin/sh

echo "=====================================BigDataLR================================="
sh local.sh 1 3 /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/test/distributed/dist_xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 0 -x auc --dis-es -p ftrl -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
#echo "=====================================BigDataFM================================="
#sh local.sh 1 1 /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 1 -x auc --dis-es -p ftrl -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
#echo "=====================================BigDataFFM==============================="
#sh local.sh 1 1 /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 2 -x auc --dis-es -p ftrl -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
