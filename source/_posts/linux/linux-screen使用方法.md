---
title: linux screen使用方法
date: 2022-08-15 20:26:26
tags:
    - linux
categories:
    - 工欲善其事
    - linux
---

使用ssh连接服务器后运行程序，如果断开ssh程序也会断开，使用screen可以解决这个问题。

## screen 安装
~~~shell
apt update
apt install screen
apt upgrade
~~~

## screen 常用命令
~~~shell
screen -S name      # 新建screen
screen -ls          # 查看screen列表
screen -r name/id   # 恢复screen
screen -X -S name/id quit     # 删除screen
~~~

## 参考资料
[screen 使用教程](https://www.jianshu.com/p/e7d1f5cc5d07?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation)
[screen 使用方法](https://blog.csdn.net/landing_guy_/article/details/124334016)