---
title: Video Graph Transformer for Video Question Answering 论文阅读
date: 2022-08-07 11:00:39
tags:
    - [Video]
    - [transformer]
    - [graph]
    - [VQA]
categories: 论文阅读
mathjax: true
---



[论文地址](https://arxiv.org/abs/2207.05342)

## Abstract

提出了Video Graph Transformer(VGT)。
1. 设计了一个dynamic graph transformer模块--通过显示捕捉(explicitly capturing)视觉对象，对象间的关系和动态来对视频进行编码，来进行复杂的时空推理(spatio-temporal reasoning)。
2. 利用解耦的(disentangled)视频和文本Transformers进行视频和文本之间的相似性比对(relevance comparison)以进行QA,而不是用纠缠不清(entangled)跨模态transformer的进行答案分类。

通过更合理(reasonable)的video encoding 和 QA solution,我们表明VGT在挑战动态关系推理的无预训练场景的任务上有更好的表现。VGT也可以从supervised cross-modal pretraining 中获益，并且所需要的数据量要小几个数量级。

[论文code地址](https://github.com/sail-sg/VGT)


## 1. Introduction
>  In particular, since 2019 [11], we have been witnessing a drastic advancement in such multi-disciplinary AI where computer vision, natural language processing as well as knowledge reasoning are coordinated for accurate decision making.

这些进步部分来自于网络规模视觉文本数据的多模态预训练的成功；部分来自于能同时对视觉和文本信息建模的统一的深度神经网络(i.e., transformer)。

尽管令人兴奋，但我们发现这种Transfomer式模型所取得的进步主要在于回答需要对视频内容进行**整体**识别或描述的问题。回答挑战现实世界视觉关系推理的问题，特别是具有视频动态的因果关系和时间关系，在很大程度上没有得到充分探索。跨模式预训练似乎很有希望。然而，它需要处理令人望而却步的大规模视频文本数据，否则性能仍然不如最先进的 (SoTA) 传统技术。

上述问题的两个可能原因:
1. 视频encoders过于简单。
    在稀疏帧上运行2D neural networks(CNNs or Transformers)、或者在短视频段上运行3D神经网络。这样的网络对视频进行了整体编码，但未能明确建模细粒度的细节，即**视觉对象之间的时空交互**。
2. VideoQA问题的表达方式是次优的(sub-optimal)。
    通常，在多选 QA 中，video、question和每个candidate answer都被附加（或融合）到一个整体令牌序列中，并馈送到跨模态 Transformer 以获得答案分类的全局表示。**由于video 和 question 比candidate长的多**，这种全局表示在消除candidate answer的歧义(disambiguating)方面很弱。
    在 open-ended QA答案被视为类索引，它们的词语义（word semantics，对 QA 很有帮助）被忽略了。

VGT从两方面解决这个问题：
1. 对于video encoder，它设计了一个dynamic graph transformer模块，该模块明确地捕获对象和关系以及它们的动态，以提高动态场景中的视觉推理。
2. 对于problem formulation，它利用单独的visual和text transformer分别对视频和文本进行编码以进行相似性（或相关性）比较，而不是使用单个跨模态transformer来融合视觉和文本信息以进行答案分类。 Vision-text通信由额外的跨模式交互模块完成。

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
$\mathcal{A}$ 可以是 $\mathcal{A}_{mc}$ 表示 multi-choice, $\mathcal{oe}$ 表示open-ended。
设计了一个video graph transformer(VGT)来执行公式中的映射 $\mathcal{F}_W$

VGT以视觉对象图为输入，结合文本信息驱动一个全局特征 $f_qv$ 来表示查询相关的视频内容。
通过语言模型（例如，BERT）获得所有候选答案的 $F^{\mathcal{A}}$。最终答案 $a^∗$ 由通过点积返回 $f^{qv}$ 和 $f^a \in F^\mathcal{A}$ 之间具有最大相似性（相关性分数）的候选答案来确定。
改模型的核心是dynamic graph transformer module(DGT)

### 3.2 Video Graph Representation
![graph construction](https://takoyaki-1258020527.cos.ap-nanjing.myqcloud.com/takoyaki/multimodal/vqa/Video%20Graph%20Transformer%20for%20Video%20Question%20Answering/graph%20construction.png)
使用预训练的 object detector提取n个RoI-aligned features 作为appearance representations $F_r = \{f_{r_i}\}^n_{i=1}$ 和他们的空间位置 $B = \{ b_{r_i}\}^n_{i=1}$, $r_i$代表了一帧中的 $i-th$ object region。
另外使用预训练的 image classification 模型获取image-level feature $F_I = \{f_{I_t}\}^{l_v}_{t=1}$

定义 linking score 来寻找同一个clip里相同的object:
$$
s_{i,j} = \psi(f^t_{r_i}, f^{t+1}_{r_j}) + \lambda \times IoU(b^t_{r_i}, b^{t+1}_{r_j}), t\in \{1,2, \cdots, l_c-1\}
$$
$\psi$ 表示余弦相似度(cosine similarity), $\lambda$ 实验中总设置为1，每个clip的第1个frame中的n个检测出的objects被设为anchor objects，相邻frame通过 greedily maximizings frame by frame联系。通过对齐一个clip中的objects，我们确保了从不同帧中生成的graphs的node 和 edge的一致性。

接下来，将object appearance $f_r$ 和 location $f_{loc}$ 表征连接并映射到 d-dimensional 空间。
$$
f_o = ELU(\phi w_o ([f_r;f_{loc}]))
$$
$[;]$ 表示特征连接 $f_{loc}$ 通过在relative coordinates上使用 $1\times 1$ 卷积获取。函数 $\phi w_o$ 表示使用参数 $w_o$ 的一个线性转换，使用 $F_o = \{f_{oi}\}^n_{i=1}$ 第 t 帧中的关系可以初始化为成对相似度:
$$
R_{t}=\sigma\left(\phi_{W_{a k}}\left(F_{o_{t}}\right) \phi_{W_{a v}}\left(F_{o_{t}}\right)^{\top}\right), \quad t \in\left\{1,2, \ldots, l_{v}\right\} \tag4
$$

简单来说，用 $G_t = (F_{ot}, R_t)$ 来表示graph的node和edge。

### 3.3 Dynamic Graph Transformer

dynamic graph transformer(DGT) 以clip-wisely $\{G_t\}^{L_v}_{t=1}$为输入，通过挖掘对详解的时间运动(temporal dynamics)和他们的空间交互(spatial interactions)输出一系列表征 $F^{DGT} \in \mathcal{R} ^{d \times k}$。
为此，我们依次操作 **a temporal graph transformer unit**, **a spatial graph conviolution unit** 和 **a hierarchical aggregation unit**。

**Temporal Graph Transformer**
![Temporal graph tansformer](https://takoyaki-1258020527.cos.ap-nanjing.myqcloud.com/takoyaki/multimodal/vqa/Video%20Graph%20Transformer%20for%20Video%20Question%20Answering/temporal%20graph%20transformer.png)
temporal graph transformer unit 将一组 $G_{in}$ 作为输入，通过node transformer(NTrans) 和 edge transformer(ETrans) 挖掘 temporal dynamics，最终输出 $G_{out}$。
回顾transformer,使用多头自注意力机制融合输入的特征 $X_{in} = \{x^t_{in}\}^l_{t=1}$:
$$
X_{\text {out }}=\operatorname{MHSA}\left(X_{i n}\right)=\phi_{W_{c}}\left(\left[h_{1} ; h_{2} ; \ldots, h_{e}\right]\right)
$$
where $\phi_{W_{c}}$ is a linear transformation with parameters $W_{c}$, and
$$
h_{i}=\operatorname{SA}\left(\phi_{W_{i_{q}}}\left(X_{\mathrm{in}}\right), \phi_{W_{i_{k}}}\left(X_{\mathrm{in}}\right), \phi_{W_{i_{v}}}\left(X_{\mathrm{in}}\right)\right)
$$
where $\phi_{W_{i_{q}}}, \phi_{W_{i_{k}}}$ and $\phi_{W_{i_{v}}}$ denote the linear transformations of the query, key, and value vectors of the $i$-th self-attention (SA) head respectively. $e$ denotes the number of self-attention heads, and SA is defined as:
$$
\operatorname{SA}\left(X_{q}, X_{k}, X_{v}\right)=\sigma\left(X_{k} X_{q}^{\top} / \sqrt{d_{k}}\right) X_{v},
$$
in which $d_{k}$ is the dimension of the key vector. Finally, a skip-connection with layer normalization (LN) is applied to the output sequence $X=L N\left(X_{\text {out }}+X_{i n}\right)$.
$X$ 可以经历更多的MHSAs 取决于transformer的层数。
In temporal graph transformer, we apply $H$ self-attention blocks to enhance the node (or object) representations by aggregating information from other nodes of the same object from all adjacent frames within a clip:
$$
F_{o_{i}}^{\prime}=\operatorname{NTrans}\left(F_{o_{i}}\right)=\operatorname{MHSA}^{(I I)}\left(F_{o_{i}}\right),
$$
in which $F_{o_{i}} \in \mathbb{R}^{l_{c} \times d}$ denotes a sequence of feature representations corresponding to object $i$ in a video clip of length $l_{c}$. 

node transformer: 它模拟单个对象行为的变化，从而推断出动态动作（例如弯腰）。此外，在某些帧处的对象遭受运动模糊或部分遮挡的情况下，它有助于改善对象的外观特征。

基于新的nodes $F'_o = \{F'_{o_i}\}^n_{i=1}$, 我们基于公式4 更新relation matrix $R$, 然后，为了显式建模时间关系动态，我们在更新的关系矩阵上应用edge transformer:
$$
\mathcal{R} = \{R_t\}^l_{t=1} \in \mathcal{R}^{l_c \times d_n}
$$
where $\mathcal{R} = \{R_t\}^l_{t=1} \in \mathcal{R}^{l_c \times d_n} (dn = n2)$ 是邻接矩阵的 row-wisely 扩充. 我们的动机是在静态帧中捕获的关系可能是虚假的(spurious)、微不足道的(trivial)或不完整的(incomplete)。edge transformer可以帮助校准错误的关系并召回丢失的关系。为简洁起见，我们将第 t 帧的时间上下文图称为 $G_{out_t} = (F'_{ot} , R'_{t})$。

**spatial graph convolution**
temporal graph transformer专注于时间关系推理。为了推理对象空间交互，我们在所有 $l_v$ 图上应用 U 层图注意力卷积:
$$
F^{′{(u)}}_o = ReLU((R′ + I)F^{′(u−1)}_o W^{(u)}),
$$
where $W^{(u)}$ is the graph parameters at the u-th layer. I is the identity matrix for skip connections. $F^zzz{′(u)}_o$ are initialized by the output node representations $F^′_o$ as aforementioned. The index t is omitted for brevity. A last skip-connection:$Fo_{out} = F^′_o + F^{′(U)}_o$ is used to obtain the final node representations.

**hierachical aggregation**
![hierachical aggregation](https://takoyaki-1258020527.cos.ap-nanjing.myqcloud.com/takoyaki/multimodal/vqa/Video%20Graph%20Transformer%20for%20Video%20Question%20Answering/Hierarchical%20Aggregation.png)
到目前为止，节点表示已经明确地考虑了对象的空间和时间交互。但这种相互作用大多是原子的(atomic)。为了将这些原子交互聚合到更高级别的视频元素中，我们采用了分层聚合策略.
首先，用一个简短的attention 将不同帧的 graph nodes聚合：
$$
f_G = \sum ^N _{i=1} \alpha_i Fo_{out_i}, \alpha = \theta (\phi W_G (Fo_out))
$$
where $\phi_{W_{G}}$ is linear transformation with parameters $W_{G} \in \mathbb{R}^{d \times 1}$. The graph representation $f_{G}$ captures a local object interactions. It may lose sight of a global picture of a frame, especially since we only retain $n$ objects and cannot guarantee that they include all the objects of interest in that frame. As such, we complement $f_{G}$ with the frame-level feature $f_{I}$ by concatenation:
$$
f_{G}=\mathrm{ELU}\left(\phi_{W_{m}}\left(\left[\phi_{W_{f}}\left(f_{I}\right) ; f_{G}\right]\right)\right)
$$


in which $\phi_{W_{m}}$ and $\phi_{W_{f}}$ are linear transformations with parameters $W_{m} \in \mathbb{R}^{2 d \times d}$ and $W_{f} \in \mathbb{R}^{2048 \times d}$ respectively. We next pool the local interactions to obtain a sequence of clip-level feature representations via:
$$
f^{\mathrm{DGT}}=\operatorname{MPool}\left(F_{G}\right)=\frac{1}{l_{c}} \sum_{t=1}^{l_{v}} f_{G_{t}}
$$
The set of $k$ clips are finally represented by $F^{\mathrm{DGT}}=\left\{f_{c}^{\mathrm{DGT}}\right\}_{c=1}^{k}$.
### 3.4 Cross-modal Interaction

To find the informative visual contents with respect to a particular text query, a cross-model interaction between the visual and textual nodes is essential. Given a set of visual nodes denoted by $X^{v}$, we integrate textual information $X^{q}=$ $\left\{x_{m}^{q}\right\}_{m=1}^{M}$ into the visual nodes via a simple cross-modal attention:
$$
x^{q v}=x^{v}+\sum_{m=1}^{M} \beta_{m} x_{m}^{q}, \quad \text { where } \quad \beta=\sigma\left(x^{v}\left(X^{q}\right)^{\top}\right),
$$
where $M$ is the number of tokens in the text query. In principle, the $X^{v}$ can be visual representations from different levels of the DGT module similar to . In our experiment, we explore perfomring the cross-modal interaction with visual representations at the object-level $\left(F_{O}\right.$ in Eqn. (3)), frame-level $\left(F_{G}\right.$ in Eqn. (12)), and clip-level $\left(F^{D G T}\right.$ in Eqn. (13)). We find that the results vary among different datasets.我们默认在DGT module (i.e. $X^{v}:=F^{\text {DGT }}$ )的输出处做交互,  因为这个阶段的节点数量要少得多，并且节点表示已经吸收了前几层的信息。For the text node $X^{q}$, we obtain them by a simple linear projection on the token outputs of a language model:
$$
X^{q}=\phi_{W_{Q}}(\operatorname{BERT}(Q)),
$$
where $W_{Q} \in \mathbb{R}^{768 \times d}$. The text query $\mathrm{Q}$ can be questions in open-end QA or QA pairs in multi-choice QA. Note that in multi-choice QA, we max-pool the obtained query-aware visual representations with respect to different QA pairs to find the one that is mostly relevant to the video.

### 3.5 Global Transformer

前面提到的 DGT 模块注重从视频剪辑中提取信息丰富的视觉线索。为了捕捉这些clip之间的时间动态，我们在跨模态交互剪辑特征（即 $F^{DGT}$）上使用了另一个 H 层transformer，并添加了可学习的正弦时间位置嵌入( learnable sinusoidal temporal position embeddings)。
$$
f^{qv} = MPool(MHSA^{(H)}(F^{DGT}))
$$
1）它保留了整体层次结构，该结构逐步驱动不同粒度的视频元素。 

2）提高了视觉和文本的特征兼容性，有利于跨模态比较。

### 3.6 Answer Prediction

To obtain a global representation for a particular answer candidate, we meanpool its token representations from $\operatorname{BERT}$ by $f^{A}=\operatorname{MPool}\left(X^{A}\right)$, where $X^{A}$ denotes a candidate answer's token representations, and is obtained in a way analogous to Eqn. (15). Its similarity with the query-aware video representation $f^{q v}$ is then obtained via a dot-product. Consequently, the candidate answer of maximal similarity is returned as the final prediction:
$$
s=f^{q v}\left(F^{A}\right)^{\top}, \quad a^{*}=\arg \max (s),
$$
in which $F^{A}=\left\{f_{a}^{A}\right\}_{a=1}^{|\mathcal{A}|} \in \mathbb{R}^{|\mathcal{A}| \times d}$, and $|\mathcal{A}|$ denotes the number of candidate answers. Additionally, for open-ended QA, we follow previous works $[60]$ and enable a video-absent QA by directly computing the similarities between the question representation $f^{q}$ (obtained in a way similar to $f^{A}$ ) and the answer representations $F^{A}$. As a result, the final answer can be a joint decision:
$$
s=f^{q v}\left(F^{A}\right)^{\top} \odot f^{q}\left(F^{A}\right)^{\top}
$$
in which $\odot$ is element-wise product. During training, we maximize the $\langle\mathrm{VQ}, \mathrm{A}\rangle$ similarity corresponding to the correct answer of a given sample by optimizing the Softmax cross entropy loss function. $\mathcal{L}=-\sum_{i=1}^{|\mathcal{A}|} y_{i} \log s_{i}$, where $s_{i}$ is the matching score for the $i$-th sample. $y_{i}=1$ if the answer index corresponds to the $i$-th sample's ground-truth answer and 0 otherwise.

### 3.7 Pretraining with Weakly-Paired Data

For cross-model matching, we encourage the representation of each video-text interacted representation $f^{q v}$ to be closer to that of its paired description $f^{q}$ and be far away from that of negative descriptions which are randomly collected from other video-text pairs in each training iteration. This is formally achieved by maximizing the following contrastive objective:
$$
\sum_{i} \log \left(\frac{\exp \left(f_{i}^{q v}\left(f_{i}^{q}\right)^{\top}\right)}{\exp \left(f_{i}^{q v}\left(f_{i}^{q}\right)^{\top}\right)+\sum_{\left(f^{q v}, f^{q}\right) \in \mathcal{N}_{i}} \exp \left(f^{q v}\left(f^{q}\right)^{\top}\right)}\right)
$$
where $\mathcal{N}_{i}$ denotes the representations of all the negative video-description pairs of the $i$-th sample. The parameters to be optimized are hidden in the process of calculating $f^{q v}$ and $f^{q}$ as introduced above. For negative sampling, we sample them from the whole training set at each iteration. For masked language modelling, we only corrupt the positive description of each video for efficiency.