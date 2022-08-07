---
title: vscode remote ssh 失败
date: 2022-08-05 15:06:39
tags:
    - vscode
    - bug
categories:
    - 工欲善其事
    - vscode
---



更新本地vscode后，服务器端的~/.vscode-server文件过期。

删除服务器端对应文件，让vscode重新下载(打开服务器的外网服务)

<details> 
    <summary>
        点击查看
    </summary> 
    <pre><code> 
[08:48:13.513] Server installation process already in progress - waiting and retrying
[08:48:14.517] Running script with connection command: ssh -T -D 58042 "192.168.31.235" bash
[08:48:14.518] Terminal shell path: C:\Windows\System32\cmd.exe
[08:48:14.900] > ]0;C:\Windows\System32\cmd.exe
[08:48:14.900] Got some output, clearing connection timeout
[08:48:14.933] "install" terminal command done
[08:48:14.933] Install terminal quit with output: 
[08:48:15.404] > 741b6048ee5c: running
[08:48:15.469] > Acquiring lock on /home/somehow/.vscode-server/bin/cd610563c72ac3f126316369server/bin/c3f126316369cd610563c75b
> 1b1725e0679adfb3/vscode-remote-lock.zhangyasheng.c3f126316369cd610563c75b1b1725e
> 0679adfb3
[08:48:15.478] > Installation already in progress...
> If you continue to see this message, you can try toggling the remote.SSH.useFloc
> k setting
> 741b6048ee5c: start
> exitCode==24==
> listeningOn====
> osReleaseId==ubuntu==
> arch==x86_64==
> tmpDir==/run/user/1017==
> platform==linux==
> unpackResult====
> didLocalDownload==0==
> downloadTime====
> installTime====
> extInstallTime====
> serverStartTime====
> 741b6048ee5c: end
    </code></pre> 
</details>

> Acquiring lock on /home/somehow/.vscode-server/bin/cd610563c72ac3f126316369

1. 删除加锁文件

   ~~~shell
   rm -rf /home/somehow/.vscode-server/bin/cd610563c72ac3f126316369
   ~~~

2. 删除 ~/.vscode-server/bin 文件夹下的文件

   ~~~shell
   rm -rf ~/.vscode-server/bin
   ~~~

删除后使用vscode remote ssh再次连接（服务器打开外网的情况下），如果服务器不能开外网可以[获取下载进程后在本地下载后上传](https://blog.csdn.net/qq_39257301/article/details/123083183?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-123083183-blog-119443079.pc_relevant_multi_platform_whitelistv3&spm=1001.2101.3001.4242.1&utm_relevant_index=3)。



## 参考

[vscode ssh连接失败](https://blog.csdn.net/myWorld001/article/details/119443079)

[vscode之ssh方式连接linux失败](https://blog.csdn.net/qq_39257301/article/details/123083183?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-123083183-blog-119443079.pc_relevant_multi_platform_whitelistv3&spm=1001.2101.3001.4242.1&utm_relevant_index=3)