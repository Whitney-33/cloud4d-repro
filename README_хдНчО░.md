# Cloud4D 复现包（Linux / NVIDIA GPU）

本包包含 [Cloud4D 官方实现](https://github.com/linjacob2/cloud4d)的完整源码，以及支持单卡、后续改 GPU 数量的训练入口 `run.py`。训练流程与官方 README 相同：Terragen 预训练 20,000 次优化更新、LES 上训练阶段 1 共 40,000 次更新、LES 上训练阶段 2 共 30,000 次更新。原始 README 见 `README.md`。源码版本见 `UPSTREAM_COMMIT.txt`。

## 1. 环境

要求 Linux、Conda、可用的 NVIDIA CUDA GPU。官方要求 Python 3.10+ 和 CUDA 12.4+ 兼容环境；下面创建独立的 Python 3.11 Conda 环境。CUDA/PyTorch wheel 须按服务器驱动实际情况选择。

```bash
cd cloud4d-repro
conda create -n cloud4d python=3.11 pip -y
conda activate cloud4d
python --version
which python
python -m pip install -U pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -U huggingface_hub
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

`requirements.txt` 保留官方训练依赖（包括 `spconv-cu124`），将 OpenCV 固定到支持当前 EXR 读取路径的 4.11.0.86，并把 W&B 日志改为本地 TensorBoard。本包现已包含 DINOv2 ViT-S/14 和 LoftUp DINOv2-S 的官方预训练权重，以及 DINOv2 的模型源码；训练初始化从 `vendor/` 本地读取，不再调用 GitHub `torch.hub`。如果服务器上仍出现 `torch.hub.load('facebookresearch/dinov2', ...)` 的错误，说明正在运行旧版代码，需要覆盖更新复现包（数据目录不用删除）。

离线权重文件为 `vendor/weights/dinov2_vits14_pretrain.pth` 与 `vendor/weights/loftup_dinov2s_pytorch_model.bin`，校验值见 `vendor/weights/SHA256SUMS`。来源分别是 [DINOv2 官方权重](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) 与 [LoftUp 官方模型](https://huggingface.co/haiwen/loftup-dinov2s)。

## 2. Data：下载后放在哪里

**只做论文的三阶段训练，下载合成数据即可**，无须下载 `real_world/`。Hugging Face 上两份压缩包分别约 1.64 GB（Terragen）和 33.10 GB（LES），需要额外留出解压空间。请在复现包根目录执行：

```bash
hf download jacoblin/cloud4d \
  synthetic/terragen.tar.gz \
  synthetic/large_eddy_simulations.tar.gz \
  unpack.py \
  --repo-type dataset --local-dir data/download
python data/download/unpack.py --subset synthetic --output data
```

下载的压缩包保存在 `cloud4d-repro/data/download/synthetic/`；**训练读取解压后的** `cloud4d-repro/data/synthetic/`。预期目录：

```text
cloud4d-repro/
├── data/
│   ├── download/
│   │   ├── unpack.py
│   │   └── synthetic/
│   │       ├── terragen.tar.gz
│   │       └── large_eddy_simulations.tar.gz
│   └── synthetic/
│       ├── terragen/
│       │   ├── perspective_1/  # 含 left_images、right_images、相机 .npy
│       │   ├── perspective_2/
│       │   ├── perspective_3/
│       │   └── volumes/       # .npz 体积标签；若实际路径不同，用参数指定
│       └── large_eddy_simulations/
│           ├── perspective_1/
│           ├── perspective_2/
│           ├── perspective_3/
│           └── volumes/       # .npz 体积标签
└── checkpoints/
```

数据卡明确给出 LES 的 `volumes/`，但未列明 Terragen 标签目录；`run.py` 会查找 `volumes/`、`5km_crop/`、`volumes/5km_crop/`。如果实际解压后不在这几个位置，使用 `--terragen-volumes /绝对路径` 指定。运行前检查：

```bash
python run.py --phase all --gpus 1 --check-only
```

它会检查三组相机目录、标定文件和 `.npz` 标签。如缺少 Terragen 标签，无法进行官方第一阶段预训练；不要把真实相机图像误当作有标签训练数据。

如果数据放在独立磁盘，例如 `/mnt/cloud4d-data/synthetic/...`，运行时加 `--data-root /mnt/cloud4d-data`；也可通过 `--terragen-images`、`--terragen-volumes`、`--les-images`、`--les-volumes` 分别覆盖路径。

## 3. 单卡训练与等效 batch

数据放好后，先检查环境与目录，再用 2 次优化更新跑通三个阶段（同时每步记录验证和 TensorBoard 图像）：

```bash
python run.py --phase all --gpus 1 --check-only
python run.py --phase all --gpus 1 --smoke-steps 2 --run-name smoke-001 --num-workers 0
```

短跑检查点单独写到 `checkpoints/smoke/smoke-001/`，阶段之间会自动接上对应短跑权重，不会覆盖正式训练的检查点。短跑只证明加载、前向、反向、验证和保存流程能跑通，不能用于判断模型质量。结束后正式运行：

```bash
python run.py --phase all --gpus 1
```

如果 Terragen 预训练已完成、LES 阶段因 EXR 解码中断，可在修好 OpenCV 后接着跑剩余两阶段（`--run-name` 与 `--smoke-steps` 必须与预训练保持一致）：

```bash
python run.py --phase finetune --gpus 1 --smoke-steps 2 --run-name smoke-001 --num-workers 0
python run.py --phase refine --gpus 1 --smoke-steps 2 --run-name smoke-001 --num-workers 0
```

三个阶段的优化更新次数都可以分别在命令行指定。例如先做原计划约十分之一步数的训练：

```bash
python run.py --phase all --gpus 1 \
  --pretrain-steps 2000 --finetune-steps 4000 --refine-steps 3000 \
  --run-name pilot-2k-4k-3k
```

该实验权重写到 `checkpoints/experiments/pilot-2k-4k-3k/`，日志写到 `runs/pilot-2k-4k-3k/`。后两阶段会自动读取本次实验的 `stage1_model_2000.pth`、`stage1_model_4000.pth`；即使最后步数不是 1000 的倍数也会保存最终权重。分别运行阶段时要复用同一个 `--run-name` 和三个步数参数。这些步数比原作者少，适合检查和初步实验，不能视作原论文完整训练复现。

默认全局有效 batch 为 4。单卡执行每卡 microbatch 1、梯度累积 4 次后更新一次；2 卡累积 2 次；4 卡累积 1 次。每阶段的 `--steps` 指**优化器更新次数**，所以仍保持 20,000 / 40,000 / 30,000。等效 batch 指每次更新的样本数一致，不保证不同 GPU 数量下逐位相同的随机性或浮点结果。显存不足时 microbatch 已是 1，无法再通过累积降低单个样本的显存占用。

## 4. 本地查看 TensorBoard

训练默认写入 `runs/<启动时间>/<阶段>/`，每个阶段有独立的事件文件。TensorBoard 的 **Scalars** 页显示训练与验证的总损失、CBH、云层厚度、LWP 和体积损失；**Images** 页显示训练与验证的投影、正交视图和高度预测图。验证和图像默认每 25 次优化更新写一次，训练标量每次更新写一次。可用 `--run-name experiment-1` 指定固定的运行名称。

在训练服务器上另开一个终端：

```bash
cd cloud4d-repro
conda activate cloud4d
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

如果要在自己的电脑浏览器里看远程服务器的 TensorBoard，先在**本地电脑**建立 SSH 隧道：

```bash
ssh -L 6006:127.0.0.1:6006 用户名@服务器地址
```

然后在本地浏览器打开 `http://127.0.0.1:6006`。训练中可以实时刷新；无需把端口公开到互联网。

## 5. 后续改 GPU 数量

只改 `--gpus`，其余参数保持：

```bash
python run.py --phase all --gpus 2
python run.py --phase all --gpus 4
```

也可以分别跑阶段：`--phase pretrain`、`--phase finetune`、`--phase refine`。默认 `--global-batch 4 --micro-batch 1`；如果以后想用 8 卡，须把 `--global-batch` 至少调到 8（这会改变论文默认的每步样本数）。可以用 `CUDA_VISIBLE_DEVICES=0,1` 控制具体用哪些卡，`--gpus` 应等于可见卡数。

输出权重：

```text
checkpoints/stage1_pretrain/stage1_model_20000.pth
checkpoints/stage1_les/stage1_model_40000.pth
checkpoints/stage2_les/stage2_model_30000.pth
```

官方训练程序保存的是模型权重，不包含优化器和学习率调度器状态。因此中断后可从已有阶段权重开始下一阶段，但不能精确恢复到同一阶段中断时的优化状态。本包未附训练好的权重或几十 GB 数据。官方仓库也没有完整的论文表格评估命令，本包覆盖公开代码能直接执行的三阶段训练流程。

## 来源

- [项目主页](https://cloud4d.jacob-lin.com/)
- [官方源码与训练命令](https://github.com/linjacob2/cloud4d)
- [官方数据及解压脚本](https://huggingface.co/datasets/jacoblin/cloud4d)
- [Accelerate 梯度累积说明](https://huggingface.co/docs/accelerate/main/usage_guides/gradient_accumulation)

源码许可证：MIT；数据许可证：CC BY 4.0。详见上游仓库和数据卡。
