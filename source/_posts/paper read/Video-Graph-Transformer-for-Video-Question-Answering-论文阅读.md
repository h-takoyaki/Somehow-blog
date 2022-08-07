---
title: Video Graph Transformer for Video Question Answering 论文阅读
date: 2022-08-07 11:00:39
tags:
    - [Video]
    - [transformer]
    - [graph]
    - [VQA]
categories: 论文阅读
---

## Abstract
提出了Video Graph Transformer(VGT)。
1. 设计了一个dynamic graph transformer模块--通过显示捕捉(explicitly capturing)视觉对象，对象间的关系和动态来对视频进行编码，来进行复杂的时空推理(spatio-temporal reasoning)。
2. 利用解耦的(disentangled)视频和文本Transformers进行视频和文本之间的相似性比对(relevance comparison)以进行QA,而不是用纠缠不清(entangled)跨模态transformer的进行答案分类。

通过更合理(reasonable)的video encoding 和 QA solution,我们表明VGT在挑战动态关系推理的无预训练场景的任务上有更好的表现。VGT也可以从supervised cross-modal pretraining 中获益，并且所需要的数据量要小几个数量级。

[论文代码地址](https://github.com/sail-sg/VGT)


## 1. Introduction
>  In particular, since 2019 [11], we have been witnessing a drastic advancement in such multi-disciplinary AI where computer vision, natural language processing as well as knowledge reasoning are coordinated for accurate decision making.

这些进步部分来自于网络规模视觉文本数据的多模态预训练的成功；部分来自于能同时对视觉和文本信息建模的统一的深度神经网络(i.e., transformer)。

尽管令人兴奋，但我们发现这种Transfomer式模型所取得的进步主要在于回答需要对视频内容进行整体识别或描述的问题。回答挑战现实世界视觉关系推理的问题的问题，特别是具有视频动态的因果关系和时间关系，在很大程度上没有得到充分探索。跨模式预训练似乎很有希望。然而，它需要处理令人望而却步的大规模视频文本数据，否则性能仍然不如最先进的 (SoTA) 传统技术。

上述问题的两个可能原因:
1. 视频encoders过于简单。
    在稀疏帧上运行2D neural networks(CNNs or Transformers)、或者在短视频段上运行3D神经网络。这样的网络对视频进行了整体编码，但未能明确建模细粒度的细节，即视觉对象之间的时空交互。
2. VideoQA问题的表达方式是次优的(sub-optimal)。
    通常，在多选 QA 中，video、question和每个candidate answer都被附加（或融合）到一个整体令牌序列中，并馈送到跨模态 Transformer 以获得答案分类的全局表示。由于video 和 question 比candidate长的多，这种全局表示在消除candidate answer的歧义(disambiguating)方面很弱。
    在 open-ended QA答案被视为类索引，它们的词语义（word semantics，对 QA 很有帮助）被忽略了。

VGT从两方面解决这个问题：
1. 对于video encoder，它设计了一个dynamic graph transformer模块，该模块明确地捕获对象和关系以及它们的动态，以提高动态场景中的视觉推理。
2. 对于problem formulation，它利用单独的visual和text transformer分别对视频和文本进行编码以进行相似性（或相关性）比较，而不是使用单个跨模态转换器来融合视觉和文本信息以进行答案分类。 Vision-text通信由额外的跨模式交互模块完成。

contuibutions:
1. 提出 Video Graph Transformer (VGT)，将 VideoQA 从浅层描述推进到深度推理。
2. 我们设计了一个动态图转换器模块，它显示了视觉推理的强度。
3. SoTA


## 2 Related Work
**Conventional Techniques for VideoQA.**
一些图方法要么构建不区分 1) 空间和时间、2) 局部和全局范围中的关系的整体图，要么在没有明确捕获时间动态的情况下在帧级别构建静态图。
对于多个对象在时空中交互的长视频来说，整体图很麻烦。此外，静态图可能导致不正确的关系（例如，拥抱与打架）或无法捕捉动态关系（例如，带走）
在这项工作中，我们将视频建模为local-to-gloval的动态视觉图，并设计graph transformer模块来显式建模对象、它们的关系和动态，以利用相邻帧中的对象和关系来校准static frame-level的虚假关系(spurious relations)。重要的是，我们还集成了强大的语言模型并探索了跨模态预训练技术，以自我监督的方式学习结构化video representation。
**Transformer for VideoQA.**
我们明确地对动态视觉推理的对象和关系进行建模，并将结构先验（或关系归纳偏差）合并到transformer架构中，以减少对数据的需求。
**Graph Transformer.**
设计和学习视频对象上的动态视觉图，并使用转换器在本地和全局范围内捕获时间动态。


## 3 Method
### 3.1 Overview
![overview](https://takoyaki-1258020527.cos.ap-nanjing.myqcloud.com/takoyaki/multimodal/vqa/Video%20Graph%20Transformer%20for%20Video%20Question%20Answering/overview%20of%20VGT.png)

$$
a^* = arg max_{a\in \mathcal{A}} \mathcal{F}_W(a|q,v, \mathcal{A})
$$
$\mathcal{A}$ 可以是 $\mathcal{A}_mc$ 表示 multi-choice, $\mathcal{oe}$ 表示open-ended。
设计了一个video graph transformer(VGT)来执行公式中的映射 $\mathcal{F}_W$

VGT以视觉对象图为输入，结合文本信息驱动一个全局特征 $f_qv$ 来表示查询相关的视频内容。
通过语言模型（例如，BERT）获得所有候选答案的 $F^{\mathcal{A}}$。最终答案 $a^∗$ 由通过点积返回 $f^{qv}$ 和 $f^a \in F^\mathcal{A}$ 之间具有最大相似性（相关性分数）的候选答案来确定。
改模型的核心是dynamic graph transformer module(DGT)

### 3.2 Video Graph Representation
![graph construction](https://takoyaki-1258020527.cos.ap-nanjing.myqcloud.com/takoyaki/multimodal/vqa/Video%20Graph%20Transformer%20for%20Video%20Question%20Answering/graph%20construction.png)

### 3.3 Dynamic Graph Transformer

### 3.4 Cross-modal Interaction

### 3.5 Global Transformer

### 3.6 Answer Prediction

### 3.7 Pretraining with Weakly-Paired Data