# -*- coding: utf-8 -*-
import os
'''
这个脚本存了一些通用的东西
比如 路径 函数等
'''
#当前目录
curDir = os.path.dirname(os.path.abspath(__file__))
#根目录
baseDir = os.path.dirname(curDir)
#静态文件目录
staticDir = os.path.join(baseDir,'Static')
#结果文件目录
resultDir = os.path.join(baseDir,'Result')
