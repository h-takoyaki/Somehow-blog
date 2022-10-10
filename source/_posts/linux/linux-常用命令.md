---
title: linux 常用命令
date: 2022-08-25 13:32:19
tags:
    - linux
categories:
    - 工欲善其事
    - linux
mathjax: true
---
## 查看删除进程

```
ps -aux | grep xxx | grep -v grep | awk '{print $2}' | xargs kill -9
```

## 查看环境变量

```
export -p
printenv
```

## tar 压缩和解压缩

c create a tar file
x unzip
z gzip
v version
f file

```
tar -czvf boot.tar.gz  ~/temp
tar -xzvf boot.tar.gz
```

## 查看文件大小

```
du -sh ./
du -sh [filename]
du -sh ./*
```

## 查看linux系统

```
uname -a #Linux查看版本当前操作系统内核信息
Linux amax 5.4.0-42-generic #46-Ubuntu SMP Fri Jul 10 00:24:02 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux

cat /proc/version #Linux查看当前操作系统版本信息
Linux version 5.4.0-42-generic (buildd@lgw01-amd64-038) (gcc version 9.3.0 (Ubuntu 9.3.0-10ubuntu2)) #46-Ubuntu SMP Fri Jul 10 00:24:02 UTC 2020

cat /etc/issue #Linux查看版本当前操作系统发行版信息
Ubuntu 20.04.1 LTS \n \l
```

## 查找文件的位置

```
find ~ -name filename
find / -name filename 2> /dev/null # 只输出正确的结果，错误结果丢到/dev/null 如果没有2就只有错误的结果。
```

## 批量移动文件

```
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
```

## 收集文件夹中的所有图片

例如：需要将这个文件夹下的所有图片的名字汇总，产生训练集。

```shell
ls -l | grep png|awk '{print $9}'| xargs -I {} echo {} >> test1.txt
```

## 移除脚本中的 `\r`

```shell
FileNotFoundError: [Errno 2] No such file or directory: 'models/tacos\r/config.yml\r'
```

Unix 体系中每行的结尾只有 `换行`, 即 `\n`, windows体系里，每行结尾是 `\n\r`

**解决方法：**

**方法一**：

```shell
# 需要先安装dos2unix
sudo dos2unix script.sh
```

**方法二：**

```shell
# sed -i 's/原字符串/新字符串/' ab.txt
# 相当于将\r 全部替换为空
sed -i 's/\r//' script.sh
```
