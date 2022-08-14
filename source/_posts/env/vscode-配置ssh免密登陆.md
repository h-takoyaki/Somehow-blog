---
title: vscode 配置ssh免密登陆
date: 2022-08-14 10:39:51
tags:
    - ssh
categories:
    - [工欲善其事, vscode]
---

1. 本地生成ssh密钥
~~~shell
ssh-keygen
~~~
可以默认按两次<kbd>enter</kbd>,
也可以自定义生成位置和passphrase，默认位置在**C:\Users\somehow\.ssh\id_rsa.pub**

2. 将公钥上传到服务器
id_rsa.pub 可以改名 id_rsa.pub.somehow

3. 将公钥传入authorized_keys文件中
~~~shell
cd ~/.ssh
cat id_rsa.pub.hzh >> authorized_keys
~~~

4. 修改vscode的config
~~~txt
Host 192.168.xx.xx
  HostName 192.168.xx.xx
  User somehow
  Port 25578
  IdentityFile C:\Users\somehow\.ssh\id_rsa
~~~