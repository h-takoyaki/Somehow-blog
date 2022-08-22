---
title: Fast Video Moment Retrieval 论文阅读
date: 2022-08-19 15:20:08
tags:
    - [Video]
    - [knowledge distillation]
    - [paper 2021]
    - [ICCV]
categories: 论文阅读
mathjax: true
---

[论文地址](https://openaccess.thecvf.com/content/ICCV2021/papers/Gao_Fast_Video_Moment_Retrieval_ICCV_2021_paper.pdf)
现有的跨模态交互耗时严重，而公共空间的效果并不好,提出了细粒度语义蒸馏框架转移知识

## Abstract
现有的VMR(video moment retrieavl)通常分为三个部分：video encoder, text encoder, cross-modal interaction module.最后一个部分跨模态交互是test-time 计算量的瓶颈。
本文作者使用cross-modal interaction common space 代替 cross-modal interaction module,将moment-query在公共空间对齐，再在common space中进行moment search。
为了学习到的空间的鲁棒性，提出了一个细粒度的语义蒸馏框架(fine-grained semantic distillation framework)将语义结构中的知识转移。具体来说，构建了一个语义角色树(semantic role tree)，将查询语句分解为不同的短语（子树）(phrases subtrees)。分层语义引导注意模块旨在在整个树中执行消息传播并产生判别特征。最后，通过matching-score蒸馏过程将重要和有区别的(discriminative)语义转移到公共空间。

## 1. Introduction
fast 的重要性
>  fast video moment retrieval (fast VMR) is in fact often necessary, since localizing the target moment is usually employed only as a part of time-critical video retrieval systems.

1. 要找的范围很大。
2. 是一个application的预处理部分。

理论上common space相较于 cross-modal interaction计算量要小很多，事实上，如果没有精心设计的跨模式交互，很难将文本查询有效地基于视频。
> How to learn a common space that can not only yield efficient moment/query features for fast VMR but also improve the discriminative ability by leveraging fine-grained semantic structures?

为了解决这个问题作者提出了使用一个细粒度的语义提取器来促进公共空间的学习。通过句子的结构来作为知识蒸馏到公共学习空间。
- 引入了快速视频时刻检索（FVMR），旨在高效准确地检索目标时刻。为此，设计了一个简单而有效的公共空间学习范式，不仅可以加快 VMR，还可以提高性能。
- 引入一种用于 FVMR 的新型细粒度语义蒸馏框架。在这里，分层语义引导注意模块旨在通过优化匹配分数蒸馏损失来利用细粒度的语义结构。
- Extensive experimental results on three popular VMR benchmarks demonstrate that our proposed method enjoys the merits of high speed and significant performance. Compared with the recent state-of-the-arts, 2DTAN , our proposed model is 40× faster and obtains5.5% absolute gains on the TACos dataset.

## 2 Related Work
现存的许多方法许多现有的 VMR 方法仅以全局方式对查询的语义信息进行编码，大部分只考虑部分语义或者使用隐式方法(implicit manenr)。

## 3 Fast Video Moment Retrieval 
![FVMR](https://takoyaki-1258020527.cos.ap-nanjing.myqcloud.com/takoyaki/multimodal/Temporal%20action%20localization/Fast%20video%20moment%20retrieval/FVMR.png)

将模型分为四个部分video encoder, text encoder, fine-grained semantic extractor,momnet-query common space.

### 3.1 Video and Text Encoders
**Video Encoder**
1. 提取proposal $P = \{p_i\}^N_{i=1}$
$$
M = \{m_1, \cdots, m_i, \cdots ,m_N\} = Encoder(\{p_i\}^N_{i=1})
$$
Encoder 可以是C3D, I3D
**Text Encoder**
sentence $S = \{s_1, \cdots, s_L\}$ 使用双向LSTM，
$$
\mathbf{w}_{1}, \mathbf{w}_{2}, \ldots, \mathbf{w}_{L}=\operatorname{BiLSTM}(S)
$$
where $w_{l}=\vec{w}_{l} \| \overleftarrow{\vec{w}}_{l}$ is the concatenation of the forward and backward hidden states of the BiLSTM for the $l$-th word. We jointly consider the addition of the beginning and the end features as the sentence feature, $\mathbf{s}=\mathbf{w}_{1}+\mathbf{w}_{L}$, where $\mathrm{s} \in \mathrm{R}^{D_{z}}$

### 3.2. Fine-grained Semantic Extractor
现阶段的许多VMR方法都忽略了句子的内在和细粒度语义结构。
使用工具[Diving Into The Relations: Leveraging Semantic and Visual Structures For Video Moment Retrieval](https://ieeexplore.ieee.org/document/9428369)构建语义角色树(semantic role tree)。
在书中将verb和noun作为叶子节点，以图为例靠近根节点的phrase能够指引下层的phrase。所以作者利用上层特征来评估下层特征的重要性。
$$
\alpha^{(j)}_k = W_\alpha(tanh(W_{top}\hat{g}^{(i)} || W_{low}\hat{g}^{(j)}_k))  \\
\hat{g}^{(i)} = \sum a^{(i)}_lg^{(i)}_l
$$
where $(i, j) \in\{(s, v),(v, n)\}$, indicating two types of consecutive hierarchy in the three-level semantic tree. $\mathbf{W}_{\alpha} \in$ $\mathrm{R}^{1 \times 2 D_{f}}, \mathbf{W}_{\text {top }} \in \mathrm{R}^{D_{f} \times D_{f}}$, and $\mathbf{W}_{\text {low }} \in \mathrm{R}^{D_{f} \times D_{f}}$ are learnable embedding matrices in the hierarchical semanticguided attention module. $\tanh (\cdot)$ is the hyperbolic tangent activation function. $\mathbf{a}^{(i)}=\operatorname{softmax}\left(\boldsymbol{\alpha}^{(i)}\right)$. With the learned importance scores $\alpha_{k}^{(j)}$, the feature of each phrase can be adaptively calculated in an attention manner:（计算短语的重要性）
$$
\begin{aligned}
&\mathbf{h}_{i}=\mathbf{b}_{i, 1} \mathbf{g}_{i}^{(v)}+\sum_{j=2}^{\mathcal{N}_{i}+1} \mathbf{b}_{i, j} \mathbf{g}_{z_{i, j}}^{(n)}, i \in\left[1, \ldots, N_{v}\right] \\
&\mathbf{b}_{i}=\operatorname{softmax}\left(\left[\boldsymbol{\alpha}_{i}^{(v)}, \boldsymbol{\alpha}_{z_{i, 1}}^{(n)}, \ldots, \boldsymbol{\alpha}_{z_{i, \mathcal{N}_{i}}}^{(n)}\right]\right)
\end{aligned}
$$
where $\mathcal{N}_{i}$ is the number of noun nodes connected with the $i$-th verb node in the semantic role tree, and $z_{i, j}$ is the corresponding index of the noun node. Finally, we incorporate all the phrase features with the global embedding:（获取每个短语的特征）
$$
\mathbf{u}=\mathbf{g}^{(s)} \odot \frac{1}{N^{(v)}} \sum_{i=1}^{N^{(v)}} \mathbf{h}_{i}
$$
where $\odot$ is the Hadamard product operator, $\mathbf{u} \in \mathrm{R}^{D_{u}}$ is the learned fine-grained semantic feature. By using Eq. (5), both the global query information and the local phrase information are leveraged, which is exploited for the following common space learning.(结合句子特征和短语特征)



### 3.3 Moment-Query Common Space Learning via Fine-Grained Semantic Distillation

**Moment-Query Common Space** 
$$
\begin{aligned}
&\mathbf{p}_{i}=\phi_{m}\left(\mathbf{m}_{i}\right)^{\top} \phi_{s}(\mathbf{s}), \\
&\mathbf{q}_{i}=\phi_{\text {fuse }}\left(\phi_{m}\left(\mathbf{m}_{i}\right) \odot \phi_{u}(\mathbf{u})\right),
\end{aligned}
$$
where $\phi_{\text {fuse }}$ is an MLP. It learns the matching score $\mathrm{q}_{i}$ by using the fused moment and fine-grained semantic features. Note that we simply use the dot product to calculate $\mathbf{p}_{i}$ for fast moment retrieval, while we additionally adopt $\phi_{\text {fuse }}$ to further consider the interaction between momen$t$ and fine-grained semantic features. Since $\mathbf{q}_{i}$ exploits the fine-grained interaction, it is served as the teacher for the following fine-grained semantic distillation.

*m moment, s sentence, u fine-grained semantic distillation*



**Video Momnt retrieval loss**

Video Moment Retrieval Loss. Because different moment proposals have different lengths, we compute the IoU score $o_{i}$ for each proposal with the ground truth moment. Similar to [65], two thresholds $o_{\min }$ and $o_{\max }$ are set to calculate the soft label $\mathbf{y}_{i}=\frac{o_{i}-o_{\min }}{o_{\max }-o_{i}}$ for the $i$-th proposal. Note that if $\mathbf{y}_{i} \leq 0$, we set $\mathbf{y}_{i}=0$, and we set $\mathbf{y}_{i}=1$ if $\mathbf{y}_{m} \geq 1$. With the soft labels, we train the video moment retrieval task by two binary cross entropy losses $\mathcal{L}_{c e}(\mathbf{p}, \mathbf{y})$ and $\mathcal{L}_{c e}(\mathbf{q}, \mathbf{y})$. Taking the former as an example:
$$
\mathcal{L}_{c e}(\mathbf{p}, \mathbf{y})=-\frac{1}{N} \sum_{i=1}^{N} \mathbf{y}_{i} \log \mathbf{p}_{i}+\left(1-\mathbf{y}_{i}\right) \log \left(1-\mathbf{p}_{i}\right)
$$
**Fine-Grained Semantic Distillation**
$$
\mathcal{L}_{d i s}(\mathbf{p}, \mathbf{q})=\mathcal{L}_{c e}\left(\sigma\left(\frac{\mathbf{p}}{T}\right), \sigma\left(\frac{\mathbf{q}}{T}\right)\right)
$$
where $T$ is a temperature hyperparameter, $\sigma$ is the softmax function. The softmax operation considers the distribution of proposal scores for knowledge distillation.

$\mathcal{T}$用作知识蒸馏的温度
$$
\mathcal{L}=\mathcal{L}_{c e}(\mathbf{p}, \mathbf{y})+\mathcal{L}_{c e}(\mathbf{q}, \mathbf{y})+\lambda \mathcal{L}_{d i s}(\mathbf{p}, \mathbf{q})
$$