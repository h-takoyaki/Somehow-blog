---
title: pathlib
date: 2022-10-04 09:24:28
tags: 
    - [python, package]
categories: 
    -[python, package]
---
将os.path 转化为 pathlib的类使用。

```python
from pathlib import Path

# 创建path类
p = Path('.')
```

os.path 和 pathlib之间的对应关系

| os and os.path               | pathlib                                       |
| ---------------------------- | --------------------------------------------- |
| `os.path.abspath`          | `Path.resolve`                              |
| `os.chmod`                 | `Path.chmod`                                |
| `os.mkdir`                 | `Path.mkdir`                                |
| `os.rename`                | `Path.rename`                               |
| `os.replace`               | `Path.replace`                              |
| `os.rmdir`                 | `Path.rmdir`                                |
| `os.remove`, `os.unlink` | `Path.unlink`                               |
| `os.getcwd`                | `Path.cwd`                                  |
| `os.path.exists`           | `Path.exists`                               |
| `os.path.expanduser`       | `Path.expanduser` and `Path.home`         |
| `os.path.isdir`            | `Path.is_dir`                               |
| `os.path.isfile`           | `Path.is_file`                              |
| `os.path.islink`           | `Path.is_symlink`                           |
| `os.stat`                  | `Path.stat`, `Path.owner`, `Path.group` |
| `os.path.isabs`            | `PurePath.is_absolute`                      |
| `os.path.join`             | `PurePath.joinpath`                         |
| `os.path.basename`         | `PurePath.name`                             |
| `os.path.dirname`          | `PurePath.parent`                           |
| `os.path.samefile`         | `Path.samefile`                             |
| `os.path.splitext`         | `PurePath.suffix`                           |

## 常用操作

```python
p = Path('/learning/test')

p.cwd().is_dir()
p.name # 'test'

# 拼接路径
Path('/').joinpath('home', 'file') # '/home/file'
Path('/') / 'home' / 'file' / 'code' # '/home/file/code'

# 解析 ~
(Path('~/lyanna').expanduser() / 'config.py').is_file()

# 上级目录
p.parent #learning
p.parent.parent #'/'
p.parents[0] # learning 数字越大越靠近根目录

# 获取文件名和后缀
f = Path('/learning/test/1.py')
f.suffix # .py
f.stem # 1

# 更改后缀和文件名
j = Path('/home/gentoo/screenshot/abc.jpg')
p.with_suffix('.png')
# PosixPath('/home/gentoo/screenshot/abc.png')
p.with_name(f'123{p.suffix}')
# PosixPath('/home/gentoo/screenshot/123.jpg')

# mkdir
Path('1/2/3').mkdir(parents=True)

#owner
p.owner()
```

## reference

[你应该使用pathlib代替os.path](https://zhuanlan.zhihu.com/p/87940289)

[pathlib--Object-oriented filesystem paths](https://docs.python.org/3/library/pathlib.html)
