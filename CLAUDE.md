# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TranAD 是一个用于多元时间序列异常检测的深度 Transformer 网络，发表于 VLDB 2022。该代码库还实现了多个基准模型用于对比实验。

## 常用命令

### 安装
```bash
pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt
```

### 数据预处理
```bash
python3 preprocess.py SMAP MSL SWaT WADI SMD MSDS UCR MBA NAB synthetic
```
预处理后的数据保存在 `processed/` 目录，每个数据集一个子文件夹。

### 运行模型
```bash
# 训练并测试
python3 main.py --model <model> --dataset <dataset> --retrain

# 仅测试（使用已保存的模型）
python3 main.py --model <model> --dataset <dataset> --test

# 使用 20% 数据训练（快速实验）
python3 main.py --model <model> --dataset <dataset> --retrain --less
```

### 可用模型
- `TranAD` - 主模型（完整版：自条件 + 对抗 + MAML）
- 消融实验模型：`TranAD_SelfConditioning`, `TranAD_Adversarial`, `TranAD_Transformer`, `TranAD_Basic`
- 基准模型：`GDN`, `MAD_GAN`, `MTAD_GAT`, `MSCRED`, `USAD`, `OmniAnomaly`, `LSTM_AD`, `Attention`, `DAGMM`, `CAE_M`
- 特殊模型：`MERLIN`（无需训练）

### 可用数据集
`synthetic`, `SMD`, `SWaT`, `WADI`, `SMAP`, `MSL`, `MSDS`, `UCR`, `MBA`, `NAB`, `energy`, `PowerSystemAnomalyDetection`

注：`energy` 与 `PowerSystemAnomalyDetection` 为单列评估数据集（仅对第 0 列 `meter_reading` / `power_load` 标注异常并统计指标）。

## 架构设计

### 核心文件结构
```
src/
├── models.py      # 所有模型定义（PyTorch nn.Module）
├── constants.py   # 超参数配置（学习率、POT阈值等）
├── parser.py      # 命令行参数解析
├── main.py        # 训练/测试入口
├── pot.py         # Peak-Over-Threshold 异常阈值检测
├── diagnosis.py   # Hit@ 和 NDCG@ 评估指标
├── merlin.py      # MERLIN 算法实现（无需训练的异常检测）
├── dlutils.py     # 深度学习工具：ConvLSTM、PositionalEncoding、Transformer 层
├── utils.py       # 可视化和数据处理工具
├── plotting.py    # 绘图功能
└── params.json    # 各数据集/模型的精细超参数覆盖
```

### 执行流程 (main.py)
1. `load_dataset()` - 从 `processed/` 加载 train/test/labels 的 .npy 文件
2. `load_model()` - 根据模型名称实例化模型（从 `src.models` 动态获取）
3. `convert_to_windows()` - 将时序数据转换为滑动窗口格式
4. `backprop()` - 模型特定的训练/推理逻辑（包含在 main.py 中）
5. `pot_eval()` - 使用 POT 方法确定异常阈值
6. 输出 Precision/Recall/F1/ROC-AUC 等指标

### 模型初始化
模型接收 `feats` 参数（特征维度），从测试集标签的 shape[1] 获取。
```python
model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])
```

### backprop() 函数
该函数（位于 main.py:76-293）包含大量 `if/elif` 分支处理不同模型的训练逻辑：
- `DAGMM` - 使用 GMM 损失
- `Attention` - 自注意力重构
- `OmniAnomaly` - VAE + KLD 损失
- `USAD` - 双自动编码器对抗训练
- `GAN` 系列 - 生成对抗网络
- `TranAD` 系列 - Transformer 编码器-解码器 + 两阶段推理

### 超参数配置
- `src/constants.py` 中的 `lr_d` 字典定义各数据集的学习率
- `lm_d` 字典定义各数据集的 POT 阈值参数
- 可通过 `src/params.json` 覆盖特定数据集+子文件的参数

## 数据格式

### 预处理后数据
`processed/<dataset>/` 下包含：
- `{file}_train.npy` - 训练数据
- `{file}_test.npy` - 测试数据
- `{file}_labels.npy` - 异常标签

### 特定数据集的文件命名
- SMD: `machine-1-1_train.npy` 等
- SMAP: `P-1_train.npy` 等
- MSL: `C-1_train.npy` 等
- UCR: `136_train.npy` 等
- NAB: `ec2_request_latency_system_failure_train.npy` 等

## 输出
- 模型检查点：`checkpoints/<model>_<dataset>/model.ckpt`
- 训练曲线：`plots/<model>_<dataset>/training-graph.pdf`