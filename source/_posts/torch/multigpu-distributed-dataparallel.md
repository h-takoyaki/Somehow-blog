---
title: 'multigpu: distributed dataparallel'
date: 2022-09-29 17:04:27
tags:
    - torch
    - torchrun
    - distributed
categories:
    - torch
    - multipgu
---
> 使用torchrun利用多GPU进行训练

- DataParallel

```
model = nn.Dataparallel(modle, device_ids=gpu_ids)
```

使用简单，但是在使用过程中会发现加速并不明显，而且会有严重的负载不均衡。这里主要原因是虽然模型在数据上进行了多卡并行处理，但是在计算loss时是统一到第一块卡再计算处理的，所以第一块卡的负载要远大于其他卡。

- distributedDataparallel

使用distributed的方法改进，和dataparallel相比分布式训练的方法能够：

1. 多进程训练，不受GIL影响(dataparallel中参数统一到第一块GPU，通信占了很大的消耗)
2. 可以进行多机多卡（dataparallel只能进行单机多卡）

DDP采用**多进程控制**多GPU，共同训练模型，一份代码会被pytorch自动分配到n个进程并在n个GPU上运行。 DDP运用Ring-Reduce通信算法在每个GPU间对梯度进行通讯，交换彼此的梯度，从而获得所有GPU的梯度。对比DP，不需要在进行模型本体的通信，因此可以加速训练。

```shell
python -m torch.distributed.launch \
       --nnodes 1 \
       --nproc_per_node=4 \
       YourScript.py
```

```txt
FutureWarning: The module torch.distributed.launch is deprecated and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun. If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See https://pytorch.org/docs/stable/distributed.html#launch-utility for further instructions

```

这种启动方式将会被torchrun代替。

- torchrun

```shell
OMP_NUM_THREADS=12               \
CUDA_VISIBLE_DEVICE=0,1,2,3     \
torchrun     --standalone       \
             --nnodes=1         \
             --nproc_per_node=4 \
              multi_quick.py
```

和之前相比local_rank,world_size由torchrun分配，要调用直接从环境中获取os.environ['local_rank']，更方便和简洁。还增加了两个新的特性：

[和distributed之间的区别](https://zhuanlan.zhihu.com/p/501632575)

[写法可以参考facebookai detr](https://github.com/facebookresearch/detr/blob/main/util/misc.py)

## 基于官方的quick_start例子更改为torchrun

讲官方的[quick_start](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)利用torchrun实现多gpu训练。

[最终代码multi_quick](https://github.com/h-takoyaki/pytorch_note/blob/main/multigpu/multi_quick.py)

- 启动命令

  ```shell
  OMP_NUM_THREADS=12               \
  CUDA_VISIBLE_DEVICE=0,1,2,3     \
  torchrun     --standalone       \
               --nnodes=1         \
               --nproc_per_node=4 \
                multi_quick.py
  ```

  OMP_NUM_THREADS 每个GPU上的CPU线程数，保证OMP_NUM_THREADS * GPU_NUM的线程总数小于总CPU线程数。没有会报一个warning自动设置为1.

  ```shell
  # 查看线程数
  cat /proc/cpuinfo | grep 'processor' | sort | uniq | wc -l
  ```

  CUDA_VISIBLE_DEVICE, 选择此程序可见的GPU。
- 获取local_rank，即当前进程在哪一个GPU上运行

  ```python
  import os
  local_rank = os.environ['local_rank']
  ```
- 让model支持分布式

  ```shell
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],output_device=local_rank)
  ```
- 讲datasets分布式地sample到不同的GPU上

  ```python
  def load_data(train_file, test_file, batch_size=32):
      train_dataset = SentenceDataset(train_file)
      test_dataset = SentenceDataset(test_file)
  
      train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  
      train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
      test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
      return train_dataloader, test_dataloader
  ```

  这里的训练集使用了torch.utils.data.distributed.DistributedSampler支持分布式，测试集是普通的写法，我们想在训练的时候分布式，测试的时候只在一张GPU上进行训练。
- evaluation

  ```python3
  if local_rank == 0:
      MSE = nn.MSELoss()
      with torch.no_grad():
           model.eval()
           for test_batch in test_dataloader:
  ```

## 一些注意点

- batchsize设置：这里的batchsize是每一个GPU上的大小原来为64，4张卡一起训练应该为16（和原来相同的效果）

- dis.barrier() 卡的训练有快有慢，有了这个可以保持同步。注意不要放到单个GPU的代码中

  ```python
  # 错误用法
  if int(os.environ['LOCAL_RANK']) == 0:
          test(test_dataloader, model, loss_fn)
          dist.barrier() # 不是local_rank的进程永远到不了，程序会暂停在这里
  ```
  
- bn层需要使用SYNbn

- 如果想要输出可以通过lock_rank,改变不同进程的logging的输出等级

  ```python
  def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
      logger = logging.getLogger(name)
      # don't log results for the non-master process
      if distributed_rank > 0:
          logger.setLevel(logging.ERROR)
          return logger
      logger.setLevel(logging.DEBUG)
      #ch = logging.StreamHandler(stream=sys.stdout)
      #ch.setLevel(logging.DEBUG)
      formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
      #logger.setFormatter(formatter)
      # logger.addHandler(ch)
  
      if save_dir:
          fh = logging.FileHandler(os.path.join(save_dir, filename))
          fh.setLevel(logging.DEBUG)
          fh.setFormatter(formatter)
          logger.addHandler(fh)
  
      return logger
  ```

  local_rank == 0输出debug信息，local_rank>0只输出error信息。
  
- 随机种子

  想要获得相同的结果常常设立随机种子。在生成数据时，如果我们使用了一些随机过程的数据扩充方法，那么，各个进程生成的数据会带有一定的同态性。

  ~~~python
  import random
  import numpy as np
  import torch
  
  def init_seeds(seed=0, cuda_deterministic=True):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
      if cuda_deterministic:  # slower, more reproducible
          cudnn.deterministic = True
          cudnn.benchmark = False
      else:  # faster, less reproducible
          cudnn.deterministic = False
          cudnn.benchmark = True
          
  
  def main():
      # 一般都直接用0作为固定的随机数种子。
      init_seeds(0)
  ~~~

  

  ~~~python
  def main():
      rank = torch.distributed.get_rank()
      # 如果使用了数据扩充的方法，可以让每个进程的seed不一样
      init_seeds(1 + rank)
  ~~~

  

## reference

[【原创】【深度】DDP系列](https://zhuanlan.zhihu.com/p/178402798)

[pytorch docs](https://pytorch.org/docs/stable/notes/ddp.html)

[Pytorch - 分布式训练极简体验](https://zhuanlan.zhihu.com/p/477073906)

[pytorch quick-start](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

[pytroch 单机多卡](https://zhuanlan.zhihu.com/p/510718081)

[pytorch 分散式训练 DistributedDataParallel](https://medium.com/ching-i/pytorch-%E5%88%86%E6%95%A3%E5%BC%8F%E8%A8%93%E7%B7%B4-distributeddataparallel-%E5%AF%A6%E4%BD%9C%E7%AF%87-35c762cb7e08)

[pytorch使用DistributedDataParallel进行多卡加速训](https://cloud.tencent.com/developer/article/1895803)
