---
title: hexo 标签和分类
date: 2022-08-05 15:32:33
tags: 
    - blog
    - 美化
categories:
    - blog搭建
---

~~~txt
---
title: hexo 标签和分类
date: 2022-08-05 15:32:33
tags: 
    - tags
    - blog
    - 美化
categories:
    - blog
---
~~~

如果你需要为文章添加多个分类，可以尝试以下 list 中的方法。
~~~txt
categories:
- [Diary, PlayStation]
- [Diary, Games]
- [Life]
~~~
此时这篇文章同时包括三个分类： PlayStation 和 Games 分别都是父分类 Diary 的子分类，同时 Life 是一个没有子分类的分类。

## 参考
[hexo Front-matte](https://hexo.io/zh-cn/docs/front-matter.html)