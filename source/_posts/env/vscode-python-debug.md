---
title: vscode python 调试配置
date: 2022-08-04 10:50:47
tags:
    - vscode 
    - debug
    - python
categories: 
    - 工欲善其事
    - vscode
---

在vscode中远程debug python的一些配置。

在运行和调试中<kbd>Ctrl+Shift+D</kbd>，添加配置。

以训练代码为例：

~~~python 
train_net.py --config-file configs/tacos.yml OUTPUT_DIR outputs/tacos
~~~

~~~json
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [

        {
            "name": "Python: train",
            "type": "python",
            "request": "launch",
            "program": "${workspaceRoot}/train_net.py", //要进行调试的代码
            "console": "integratedTerminal",
            "justMyCode": true,
            "args": ["--config-file", "configs/tacos.yml",  "OUTPUT_DIR", "outputs/tacos"],//调试代码时候的一些参数设置
            "python": "/root/miniconda3/envs/vlg/bin/python" //想要使用的python环境
        }
    ]
}
~~~

