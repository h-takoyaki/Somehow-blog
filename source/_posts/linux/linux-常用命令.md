---
title: linux 常用命令
date: 2022-08-25 13:32:19
tags:
    - linux
categories:
    - 工欲善其事
    - linux
---

## 查看删除进程
~~~
ps -aux | grep xxx | grep -v grep | awk '{print $2}' | xargs kill -9
~~~

## 查看环境变量
~~~
export -p
printenv
~~~

## tar 压缩和解压缩
c create a tar file
x unzip
z gzip
v version
f file
~~~
tar -czvf boot.tar.gz  ~/temp
tar -xzvf boot.tar.gz
~~~

## 查看文件大小
~~~
du -sh ./
du -sh [filename]
du -sh ./*
~~~

## 查看linux系统
~~~
uname -a #Linux查看版本当前操作系统内核信息
Linux amax 5.4.0-42-generic #46-Ubuntu SMP Fri Jul 10 00:24:02 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux

cat /proc/version #Linux查看当前操作系统版本信息
Linux version 5.4.0-42-generic (buildd@lgw01-amd64-038) (gcc version 9.3.0 (Ubuntu 9.3.0-10ubuntu2)) #46-Ubuntu SMP Fri Jul 10 00:24:02 UTC 2020

cat /etc/issue #Linux查看版本当前操作系统发行版信息
Ubuntu 20.04.1 LTS \n \l
~~~

## 查找文件的位置
~~~
find ~ -name filename
find / -name filename 2> /dev/null # 只输出正确的结果，错误结果丢到/dev/null 如果没有2就只有错误的结果。
~~~

## 批量移动文件
~~~
(base) somehow@amax:~/tmp/carla$ ls
carla        CHANGELOG      Engine  ImportAssets.sh                Plugins    Tools
CarlaUE4     Co-Simulation  HDMaps  LICENSE                        PythonAPI  VERSION
CarlaUE4.sh  Dockerfile     Import  Manifest_DebugFiles_Linux.txt  README
(base) somehow@amax:~/tmp/carla$ ls -l | grep -v carla | awk '{print $9}' | xargs -I {} mv {} carla
(base) somehow@amax:~/tmp/carla$ ls
carla
(base) somehow@amax:~/tmp/carla$ cd carla
(base) somehow@amax:~/tmp/carla/carla$ ls
CarlaUE4     CHANGELOG      Dockerfile  HDMaps  ImportAssets.sh  Manifest_DebugFiles_Linux.txt  PythonAPI  Tools
CarlaUE4.sh  Co-Simulation  Engine      Import  LICENSE          Plugins                        README     VERSION
~~~