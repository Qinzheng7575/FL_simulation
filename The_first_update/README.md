## 环境

- python 3.7.12
- torch 1.4.0
- torchvision 0.5.0
- syft 0.2.4

## 文件结构

update_simu.py 是主程序，functions_for_trans.py 为增量上报信道仿真模拟时，用到的函数，加以区分出来。

## 代码结构

按照 FL 的运行逻辑，如下几个步骤:

1. 初始化 ue`init_ue()`,和 BS`BS_model`，
2. 加载、分发数据(`distribute()`)，开始本地训练(`ue.train()`)，
3. BS 聚合模型(`aggregate`)并测试(`test()`)，在聚合模型的时候，模拟上传的模型类别（完整模型上传或者模型基础量上传）。随后分发模型到 ue
4. 进入下一轮
