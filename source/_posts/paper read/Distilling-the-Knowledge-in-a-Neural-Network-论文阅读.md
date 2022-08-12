---
title: Distilling the Knowledge in a Neural Network 论文阅读
date: 2022-08-11 14:56:19
tags:
    - [knowledge distillation]
    - [paper 2015]
categories: 论文阅读
mathjax: true
---

## 参考
[论文地址](https://arxiv.org/abs/1503.02531)
[同济子豪兄](https://www.bilibili.com/video/BV1N44y1n7mU/?spm_id_from=333.788&vd_source=f9c721877addf532afef40a450bacd1d)

## Abstract
多模型集成能够提升学习性能，但是这种方法不仅繁琐而且计算成本昂贵。Caruana and his collaborators发现可以将集成(ensemble)中的知识(knowledge)压缩到单个模型中。这种方法更好部署，作者在这种方法的基础上继续研究，在MNIST手写数据集和商用语音识别模型上都有很好的提升。提出了一种新的模型集成范式(one or more full models and many specialist models)，与混合的专家模型(mixture of experts)不同，这些专家模型可以快速并行地训练。

## 1 Introduction
许多昆虫幼年时期的形态便于营养获取，成虫时期便于迁徙和繁殖，不同形态满足不同要求。但我们在大规模机器学习过程中，训练和部署往往用的是相似的模型，训练阶段我们需要提取更多的数据结构（可以是单个超大模型，也可以是多个模型集成），部署阶段需要一定实时性。当我们使用大模型获取知识后再想办法迁移到小模型上，也就是蒸馏(distillation)。

如果将“知识”定义为训练模型的学习到的参数值，那么knowledge是很难迁移到小模型上的。**将knowledge定义为输入向量到输出向量的一个映射(mapping from input vectors to output vectors)**大模型计算交叉熵损失函数(maximize the average log probability of the correct answer),它的副产品(side-effect)就是给每一个类别赋值，非正确类别预测出的相对概率的大小包含了重要的信息。e.g. BMW、垃圾车、胡萝卜，预测BMW的时候胡萝卜的等分应该低于垃圾车。

需要模型在测试集和真实任务上泛化，要定义和量化学到的知识。

一种明显的方法就是直接用cumbersome model的“soft targets”作为small model的 label 进行训练，这样我们可以使用"transfer"数据集，如果教师模型的soft tagets熵(entropy)很高，不同类型之间的差异很大，可以解析出来的知识更多。

以MNIST手写数据集为例子，一个好的模型常常给correct answer 一个非常高的置信度，而其他的非常低，比如对于识别7 那么他给2和3的置信度为 $10^{-6} 和10^{-7}$ 这说明了这个数字更像7，但是没说明是更不像2还是更不像3，为了解决这个问题前人直接用logits(the inputs to the final softmax)而不是softmax后得到的概率作为知识，这篇文章中使用了"temporature"这个概念在蒸馏的过程中增大t，可以更soft。

## 2 Distillation
神经网路使用"softmax"获取每一类的概率 $q_i$,
$$
q_i = \frac{exp(z_i/T)}{\sum_{j}{exp(z_j/T)}}
$$
> where T is a temperature that is normally set to 1. Using a higher value for T produces a softer probability distribution over classes.

$z_i$ 是logits，当T=1的时候就是我们常用的softmax。

![模型架构](https://takoyaki-1258020527.cos.ap-nanjing.myqcloud.com/takoyaki/knowledge%20distillation/Distilling%20the%20Knowledge%20in%20a%20Neural%20Network/structure.png)
soft loss只在训练的过程中用。soft loss来自老师，hard loss 来自label。

### 2.1 Matching logits is a special case of distillation

直接用logits 是temperature无限高，假设对于不同样本的logits的均值为0的一个特例
$$
\frac{\partial C}{\partial z_{i}}=\frac{1}{T}\left(q_{i}-p_{i}\right)=\frac{1}{T}\left(\frac{e^{z_{i} / T}}{\sum_{j} e^{z_{j} / T}}-\frac{e^{v_{i} / T}}{\sum_{j} e^{v_{j} / T}}\right)
$$
If the temperature is high compared with the magnitude of the logits, we can approximate:（泰勒展开公式）
$$
\frac{\partial C}{\partial z_{i}} \approx \frac{1}{T}\left(\frac{1+z_{i} / T}{N+\sum_{j} z_{j} / T}-\frac{1+v_{i} / T}{N+\sum_{j} v_{j} / T}\right)
$$
If we now assume that the logits have been zero-meaned separately for each transfer case so that $\sum_{j} z_{j}=\sum_{j} v_{j}=0$ Eq.3 simplifies to:
$$
\frac{\partial C}{\partial z_{i}} \approx \frac{1}{N T^{2}}\left(z_{i}-v_{i}\right)
$$
但是实际上temperature过高会导致噪声的出现。



---

后面是一些实验

## 3 Preliminary experiments on MNIST

## 4 Experiments on speech recognition

## 5 Training ensembles of specialists on very big datasets