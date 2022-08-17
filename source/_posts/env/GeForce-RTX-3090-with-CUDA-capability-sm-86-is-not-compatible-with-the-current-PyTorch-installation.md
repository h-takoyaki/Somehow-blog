---
title: >-
  GeForce RTX 3090 with CUDA capability sm_86 is not compatible with the current
  PyTorch installation.
date: 2022-08-17 16:22:05
tags:
    - pytorch
    - bug
categories:
    - [env, pytorch]
---

代码运行时报错
> NVIDIA GeForce RTX 3090 with CUDA capability sm_86 is not compatible with the current PyTorch installation. The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 compute_37.

RTX 算力为 sm_86, cuda 版本要在11.0以上

重装。:)