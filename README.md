## 环境

- python 3.7.12
- torch 1.4.0
- torchvision 0.5.0
- syft 0.2.4

## 文件结构：

FL_simu.py 是真正运行的代码，剩余代码是为了我方便对照留下的

## 代码架构

按照 FL 的运行逻辑，如下几个步骤

1. 初始化 ue`init_ue()`,BS`BS_model`，
2. 加载、分发数据(`alloc()`)，开始本地训练(`ue.train()`)，
3. 每训练若干 epoch（聚合频率`train_args['aggre_interval']`），就将 model 进行上传操作(`tranport()`)，**传输函数待完善**
4. BS 聚合模型(`aggregate`)，测试(`test()`)，分发
5. 进入下一轮
