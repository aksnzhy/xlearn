#! /bin/sh
#
# stop_xlearn.sh
# Copyright (C) 2017 wangxiaoshu <2012wxs@gmail.com>
#
# Distributed under terms of the MIT license.
#


ps -ef | grep xlearn_train | awk '{ print $2 }' | sudo xargs kill -9
