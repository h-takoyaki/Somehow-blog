---
title: conda 环境迁移
date: 2022-08-16 15:00:41
tags:
    - conda
    - env
categories:
    - [工欲善其事,conda]
---

## 本地克隆
~~~shell
conda create -n py377 --clone py37
~~~
将py37克隆一份命名为py377

## 同平台间迁移
使用conda-pack打包，打包完毕后迁移。
1. 下载conda-pack
~~~shell
pip install conda-pack
~~~

2. python环境打包
~~~shell
conda pack -n py37 -o py37.tar.gz
~~~
会在当前目录生成py37.tar.gz

3. 传输

4. 解压
~~~shell
mkdir py37
tar -zxvf py37.tar.gz -C py37
~~~
将py37放到conda/env/py37，如果不知道具体位置可以用
~~~shell
conda env list
~~~
查看。

## 不同平台
不同平台不能直接打包使用，但是可以生成environment.yml方便对方安装。
~~~shell
conda activate py37 # 激活你的环境
conda env export > environment.ymal #生成.yml文件

conda env create -f environment.ymal # 基于.yml生成环境
~~~