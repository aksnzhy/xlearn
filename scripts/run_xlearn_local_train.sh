#echo "====================================SmallData LR==================================="
#/Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/demo/classification/criteo_ctr/small_train.txt -v /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/demo/classification/criteo_ctr/small_test.txt -s 0 -x auc --dis-es
#echo "====================================SmallData FM==================================="
#/Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/demo/classification/criteo_ctr/small_train.txt -v /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/demo/classification/criteo_ctr/small_test.txt -s 1 -x auc --dis-es
#echo "====================================SmallData FFM=================================="
#/Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/demo/classification/criteo_ctr/small_train.txt -v /Users/xiaoshuwang/documents/oneflow/opensource/xlearn/demo/classification/criteo_ctr/small_test.txt -s 2 -x auc --dis-es

echo "=====================================BigDataLR================================="
/Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 0 -x auc --dis-es -p ftrl -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
#echo "=====================================BigDataFM================================="
/Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 1 -x auc --dis-es -p ftrl -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
#echo "=====================================BigDataFFM==============================="
/Users/xiaoshuwang/documents/oneflow/opensource/xlearn/xlearn_build/xlearn_train ~/documents/data/libffm_toy/criteo.tr.r100.gbdt0.ffm -v ~/documents/data/libffm_toy/criteo.va.r100.gbdt0.ffm -s 2 -x auc --dis-es -p ftrl -alpha 5e-2 -beta 1.0 -lambda_1 5e-5 -lambda_2 15.0 --no-norm
