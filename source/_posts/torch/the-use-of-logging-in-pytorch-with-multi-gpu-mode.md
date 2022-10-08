---
title: the use of logging in pytorch with multi-gpu mode
date: 2022-10-08 09:21:46
tags:
	- [torch, distributed]
	- logging
categories:
	- torch
	- logging
---
> 使用logging 记录日志，方便之后对实验过程(结果)的查看。可以设定logger的level，指定主进程输出。使用过程中有一些问题，主要来自于toch.distributed.run调用的forwar函数中本身也是用了logging，可能会导致冲突。

[github](https://github.com/h-takoyaki/pytorch_note/tree/main/multigpu)

[logging 代码](https://github.com/h-takoyaki/pytorch_note/blob/main/multigpu/lib/utils/logger.py)

## 通过handler设置输出终端还是输出特定文件

```python
suffix 
if show_terminal:
    # 输出终端
    ch = logging.StreamHandler(stream=sys.stdout)
    # ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

if save_dir:
    # 输出指定文件
    local_time = time.strftime("%m%d-%H", time.localtime())
    filename = local_time + suffix
    # mkdir
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    fh = logging.FileHandler(save_dir / filename)
    # fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
```

## 通过设定level对不同的进程的输出进行设定

```python
if local_rank > 0:
    logger.setLevel(logging.ERROR)
    return logger
logger.setLevel(logging.DEBUG)
```

在0号GPU(local_rank == 0)上跑的进程设置为主进程，输出等级设置为 `DEBUG`，其他GPU上的进程设置为 `ERROR`，这样在程序运行过程中必要信息只会输出一次。

train_logger test_logger的创建要在循环开始之前，否则单条语句会重复输出

```python
main_logger = setup_logger(f'logger.main.{local_rank}', True, save_path, local_rank)
train_logger = setup_logger(f'logger.train.{local_rank}',
                            True,
                            save_path,
                            local_rank)
test_logger = setup_logger(f'logger.test.{local_rank}',
                            True,
                            save_path,
                            0)

main_logger.info(f'Start Training')
synchronize()
for t in range(args.epochs):
    main_logger.info(f"Epoch {t+1} / {args.epochs}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, train_logger)
    if is_main_process():
        test(test_dataloader, model, loss_fn, test_logger)
    synchronize()
```

## 遇到的bug

> 在调用生成的logger后会在终端重复输出一次formatter为标准的信息(不管设定的是输出到终端、文件还是都输出)。

**原因：**在调用 `torch.nn.parallel.DistributedDataParallel`执行 `forward`函数的过程中调用了

```python
if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
    logging.info("Reducer buckets have been rebuilt in this iteration.")
    self._has_rebuilt_buckets = True
```

会使用logging进行初始化。

> 用logging.info等命令输出过日志的时候，logging会自动的（偷偷摸摸的）给你创建一个streamhandler

这导致了重复输出。

**解决办法：**在输出前logger.parent = None

```python
test_logger.parent = None
test_logger.info(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
```

## reference

[python logging](https://docs.python.org/3/library/logging.html)

[使用DistributedDataParallel和logging导致logger重复输出的bug](https://blog.csdn.net/weixin_42619365/article/details/121754424)

[关于logging的那些坑](https://www.cnblogs.com/telecomshy/p/10630888.html)
