---
layout: post/blog
title: hexo markdown公式显示错误
date: 2022-08-08 21:05:41
tags:
    - [blog]
    - [bug]
categories:
    - blog搭建
mathjax: true
---

## 原因
Hexo 默认使用 hexo-renderer-marked 引擎渲染网页，该引擎会把一些特殊的 markdown 符号转换为相应的 html 标签

## 解决方法
1. 将渲染引擎 hexo-renderer-marked 修改为 hexo-renderer-kramed
~~~shell
npm uninstall hexo-renderer-marked --save
npm install hexo-renderer-kramed --save
~~~

2. 内联公式仍然有bug，修改 blog\node_modules\kramed\lib\rules\inline.js
line11
~~~json
  //escape: /^\\([\\`*{}\[\]()#$+\-.!_>])/,
  escape: /^\\([`*\[\]()#$+\-.!_>])/,
~~~
line21
~~~json
  //em: /^\b_((?:__|[\s\S])+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
  em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
~~~

3. 如果启用了Next主题，修改主题目录 blog\themes\next\_config.yml
~~~json
# Math Formulas Render Support
math:
  # Default (true) will load mathjax / katex script on demand.
  # That is it only render those page which has `mathjax: true` in Front-matter.
  # If you set it to false, it will load mathjax / katex srcipt EVERY PAGE.
  per_page: true

  # hexo-renderer-pandoc (or hexo-renderer-kramed) required for full MathJax support.
  mathjax:
    enable: true #此处开启mathjax
    # See: https://mhchem.github.io/MathJax-mhchem/
    mhchem: false
~~~

4. 在文章的Front-matter里打开mathjax开关
~~~shell
---
layout: post/blog
title: hexo markdown公式显示错误
date: 2022-08-08 21:05:41
tags:
    - [blog]
    - [bug]
categories:
    - blog搭建
mathjax: true
---
~~~

4. 重启hexo
~~~shell
hexo clean
hexo g
hexo s
~~~

## 参考
[hexo next主题解决无法显示数学公式](https://blog.csdn.net/yexiaohhjk/article/details/82526604)