---
title: 'relative import: meet ''MoudleNotFoundError'' when importing your self code'
date: 2022-10-06 10:34:44
tags:
    - python
    - bug
    - import
categories:
    - [python, bug]
---
想要将部分代码整合到 `tools`中但是它们调用了 `lib`中的代码，使用相对路径import的时候会产生MoudleNotFoundError的错误。最终使用方案三解决（能够使代码呈现出想要的结构）将 `lib`的路径加入到 `sys.path`中。

**file structure**:

```shell
.
├───lib
│   ├───core
│   │   │   core_function.py
│   │   │   __init__.py
│   │   │
│   │   ├───subcore
│   │   │   │   subcore_function.py
│   │
│   └───models
│       │   model_x.py
│       │   __init__.py
│
└───tools
    │   train.py
    │   _init_path.py

```

**python中的包能否被找到，其实是从 `sys.path`出发后能否抵达。**假设我们的 `sys.path`包含 `'/home/somehow/import_test/tools/../lib'`那么当我们 `from models.model_x import xxx`就能找到 `'/home/somehow/import_test/tools/../lib.models.model_x.xxx'`,所以我们只要确定自己的包能否被 `sys.path`找到就行。举一个反例：假设我们的 `sys.path`只包含 `'/home/somehow/import_test/tools'`，由于 `tools`下是没有 `lib.models.model_x.xxx`，所以 `from models.model_x import xxx`报错。

[github](https://github.com/h-takoyaki/pytorch_note/tree/main/relative_import)

**main trick**:

add the path of `lib` into `sys.path`

```python
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, str(path))
```

> **Just knowing what directory a file is in does not determine what package Python thinks it is in.**
>
> When a file is loaded, it is given a name (which is stored in its `__name__` attribute).
>
> - If it was loaded as the top-level script, its name is `__main__`.
> - If it was loaded as a module, its name is [ the filename, preceded by the names of any packages/subpackages of which it is a part, separated by dots ], for example, `package.subpackage1.moduleX`.
>
> [from **Script vs. Module**](https://stackoverflow.com/a/14132912/15554546)

**Solutions:**

1. `python -m package.subpackage1.moduleX`. The `-m` tells Python to load it as a module, not as the top-level script.
2. put `myfile.py` *somewhere else* – *not* inside the `package` directory – and run it. If inside `myfile.py` you do things like `from package.moduleA import spam`, it will work fine.
3. ⭐️[researchmm/2D-TAN](https://github.com/researchmm/2D-TAN) add `./tools/../lib` into `sys.path`

## reference

[researchmm/2D-TAN](https://github.com/researchmm/2D-TAN)

[**Script vs. Module**](https://stackoverflow.com/a/14132912/15554546)
