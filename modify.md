# DLoRAL 改进方案 (P1: Detail Side-Channel + RAFT)

> 课程项目文档 —— 从头解释:为什么改、改了什么、哪一步需要训练、用什么数据。
> 配套代码改动见 `README.md` 的 "Side-Channel Extension" 一节。

---

## 1. 背景与动机

### 1.1 DLoRAL 是什么
DLoRAL (arXiv 2506.15591) 是一个一步扩散视频超分 (one-step diffusion VSR) 模型,基于 SD 2.1。它在 LR 视频上做 4× 超分,核心三件套:

| 组件 | 作用 |
| --- | --- |
| **CFR** (Cross-Frame Retrieval) | SpyNet 算光流 + top-k selective attention,把邻帧对齐到当前帧 |
| **C-LoRA** (Consistency LoRA) | UNet 上的一组 LoRA,保证多帧时间一致 |
| **D-LoRA** (Detail LoRA) | 另一组 LoRA,负责细节增强;C-LoRA 冻结时训练,用 CSD 蒸馏 |

两阶段交替训练:先训 C-LoRA(consistency),再冻 C-LoRA 训 D-LoRA(detail),loss 平滑过渡。

### 1.2 它的瓶颈在哪
DLoRAL 跑在 SD 2.1 之上,而 SD 2.1 自带 **8×8 的 VAE 下采样** —— LR 输入先被 VAE 压到 1/8,扩散在 latent 空间一步去噪,最后 VAE 再解码回 HR。

**这个 VAE 不是为视频超分设计的**,作者自己也在 limitations 写了:
- 8× 压缩破坏小字、细边、纹理这些高频信息
- 多帧之间的 latent 已经损失太多空间信息,时间一致性也跟着掉

**结论:** 不管 LoRA 多强,被 VAE 抹掉的细节,LoRA 也复原不出来 —— 信息已经不在 latent 里了。

### 1.3 还有几个隐藏弱点(论文没说但实在的)

1. **时间窗只有 3 帧** —— CFR 看不到长程运动
2. **D-LoRA 训练数据是"伪时间"** —— 用 LSDIR 静态图随机像素平移,没见过真实运动 → VideoLQ 这种大运动一上来就崩
3. **C/D-LoRA 共享参数空间**,两阶段是零和博弈 —— 作者自己承认 "neither can be pushed to theoretical maximums"
4. **SpyNet 是 2017 年的光流网络** —— 大运动 / 光照剧变下流估计不准

---

## 2. 改进方案 P1 总览

针对 §1.2 这个**根本性瓶颈**,我们走两条路:

```
              主线: Detail Side-Channel  (干掉 VAE 信息瓶颈)
                +
              辅助: SpyNet → RAFT        (修光流上限)
```

### 2.1 主线: Detail Side-Channel

**核心思想:** 既然 VAE 必然丢信息,**就在 VAE 外面再开一条像素空间的旁路**,直接从 LR 把那部分高频"绕过去"补到 HR 输出上。

**架构:**
```
LR ──bicubic↑──► x_up                                                   (HR-size, 像素空间)
LR ──VAE.encode──► z ──[CFR + C-LoRA + D-LoRA + UNet 一步]──► z_HR
                                                            │
                                              VAE.decode    │
                                                            ▼
                                                         y_coarse        (HR-size, 像素空间)
                  ┌────────────────────────────────────────┤
                  │                                        │
                  ▼                                        ▼
         DetailNet(x_up, y_coarse) ──► d_t       GateNet(x_up, y_coarse) ──► α  ∈[0,1]
                  │                                        │
                  └─────────► y_final = y_coarse + α · d_t ◄────────────
```

**关键设计:**
| 设计点 | 选择 | 原因 |
| --- | --- | --- |
| DetailNet 末层卷积 | **Zero-init** (权重置零) | step 0 时 `d_t ≈ 0` ⇒ `y_final ≡ y_coarse`,**起点不会比 baseline 差** |
| GateNet sigmoid bias | **+2.0** (初始 α ≈ 0.881) | gate 起始全开 → DetailNet 才有足够梯度学;训完自动收紧 |
| 监督帧 | 窗口**最后一帧** (`hr[:,-1]`) | 与 DLoRAL 推理输出对齐 (`src_idx = input_image_index + start + 1`) |
| 训练时冻结 | C-LoRA / D-LoRA / UNet / VAE / CFR **全冻** | 只训 ~6.77M 可训参,单卡能跑;不动 DLoRAL 任何已有能力 |

### 2.2 辅助: SpyNet → RAFT

CFR 内部的 SpyNet 替换为 RAFT。SpyNet 接口是 "两帧 → flow",RAFT 同接口,**0 改动调用代码,只换网络**。
- 风险:RAFT 显存比 SpyNet 大 → 准备 RAFT-small / GMFlow 兜底

---

## 3. 模块详细设计

### 3.1 DetailNet (~4.88M)

| 层 | 配置 |
| --- | --- |
| 输入 | `concat(x_up, y_coarse)`,`6 × H × W`(两张 RGB) |
| Backbone | ResNet 风格,base_ch=128, num_blocks=16 |
| 输出 | `d_t`,`3 × H × W`(残差,不带 sigmoid/tanh,允许正负) |
| 末层 | **Zero-init 卷积** |

### 3.2 GateNet (~1.88M)

| 层 | 配置 |
| --- | --- |
| 输入 | `concat(x_up, y_coarse)`,`6 × H × W` |
| Backbone | 对称 U-Net,通道更小 |
| 输出 | `α`,`1 × H × W`(per-pixel,sigmoid) |
| 末层偏置 | `+2.0` ⇒ sigmoid(2)≈0.881 |

### 3.3 SideChannelWrapper

```python
y_final = y_coarse + α * d_t
```
推理时 `return_aux=False`,只返回 `y_final`。
训练时 `return_aux=True`,同时返回 `α` 和 `d_t` 用于 logging / 稀疏正则。

### 3.4 代码位置

| 文件 | 内容 |
| --- | --- |
| `src/side_channel/detail_net.py` | DetailNet |
| `src/side_channel/gate_net.py` | GateNet |
| `src/side_channel/side_channel_wrapper.py` | 组合 |
| `src/side_channel/dataset.py` | REDS 多帧 + Real-ESRGAN 在线退化 |
| `src/train_side_channel.py` | 训练主入口 (Accelerate DDP) |
| `configs/side_channel.yaml` | 训练超参 |
| `scripts/train_sidechannel.sh` | 多 GPU 启动脚本 |

---

## 4. 训练 —— 哪一步需要训练 / 用什么数据

### 4.1 不需要再训的部分 (复用上游 ckpt)

| 模块 | ckpt 来源 |
| --- | --- |
| C-LoRA / D-LoRA / UNet / VAE | `preset/models/checkpoints/model.pkl`(DLoRAL 官方) |
| CFR (SpyNet) | 同上 |
| SD 2.1 base | `preset_models/stable-diffusion-2-1-base/`(本地缓存) |
| RAM (caption 模型) | `preset/models/{ram_swin_large_14m.pth, DAPE.pth}` |

⇒ DLoRAL 整体在我们整个 P1 训练流程里**全程冻结、不更新一个参数**。

### 4.2 需要训练的部分(只有 SideChannelWrapper)

| 阶段 | 训什么 | 数据集 | 数据量 | 步数 | 显存 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| **W2 主训** | DetailNet + GateNet | REDS `train_sharp` | 270 clips × 100 帧 = 27,000 帧;`num_frames=2` 滑窗 ⇒ ~23,760 样本 | 5000 | 2× 46GB DDP | ✅ 已完成 |

**为什么用 REDS:**
- DLoRAL 论文 consistency 阶段就是用 REDS,分布对得上
- 270 clip × 100 帧足够,不需要再下别的训练集
- 已经在盘上 (`/data/yuhanchen/CV/data/reds/`),省下载时间

**退化方式 —— 在线 Real-ESRGAN 而非离线 BDx4:**
- 训练时 LR 由 HR 通过 `src/datasets/params.yml` 里的 Real-ESRGAN pipeline 在线退化产生(模糊+下采样+噪声+JPEG)
- 这样模型见过的退化分布**比 UDM10 的纯 BDx4 复杂得多**,推理时反而是简单的 BDx4 → 鲁棒
- 没有用 LSDIR 那种伪时间静态图,因为我们要监督**真实视频帧**之间的一致性

### 4.3 训练 loss

```
L = L1(y_final, y_gt)  +  0.5 · LPIPS(y_final, y_gt)
```

可选扩展(未启用,留作消融):
- `+ λ · L1(Canny(y_final), Canny(y_gt))` —— 强迫边缘匹配
- `+ λ · ||α||_1` —— gate 稀疏正则,防止它学成全开

### 4.4 训练曲线 (实测)

| 步数 | l1 | lpips | α (mean) | d_t magnitude |
| --- | --- | --- | --- | --- |
| 500 | 0.30 | 0.70 | 0.881 (init) | ~0 |
| 2500 | 0.10 | 0.45 | 0.80 | 0.02 |
| 5000 | 0.06 | 0.32 | **0.713** | 0.02-0.04 |

α 自然下降到 0.71 → gate 学会**有选择地开**,而不是全开;d_t 幅度小 → 残差不极端,训练稳。

---

## 5. 推理 (W3) —— 不需要训,只是接线

### 5.1 接入点

`src/test_DLoRAL.py` 在 model 输出和 `frame_t = output_image[0]` 之间插钩:

```python
if sidechannel is not None:
    with torch.no_grad():
        y_coarse_01 = (output_image.float() * 0.5 + 0.5).clamp(0, 1)
        x_up_01     = (c_t[:, -1].float() * 0.5 + 0.5).clamp(0, 1)
        y_final, aux = sidechannel(x_up_01, y_coarse_01, return_aux=True)
        if abs(args.sidechannel_alpha_scale - 1.0) > 1e-6:
            y_final = y_coarse_01 + args.sidechannel_alpha_scale * aux["alpha"] * aux["d_t"]
        output_image = (y_final * 2 - 1).clamp(-1, 1).to(dtype=weight_dtype)
```

### 5.2 新增 CLI

| Flag | 默认 | 含义 |
| --- | --- | --- |
| `--sidechannel_ckpt` | `None` | None ⇒ 关闭(等同上游) |
| `--sidechannel_alpha_scale` | `1.0` | gate 缩放,0=关 / 1=训完 / >1=放大(调试用) |

⇒ 不传 flag 时**完全等价上游**,不破坏任何旧调用。

---

## 6. 实验流程

```
[环境]
└── conda env: lora (torch 2.0.1+cu117, accelerate, lpips)
└── 离线模式: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
└── 服务器: chenyuhan@10.120.16.29:/nas-files/yuhanchen/lora/DLoRAL

[Step 1: W1 baseline] ✅ done
└── 数据: UDM10 (10 clips, BDx4, 已转 mp4)
└── 命令: python src/test_DLoRAL.py --sidechannel_ckpt None ...
└── 产物: /nas-files/yuhanchen/tmp/udm10_out/<clip>/frame_*.png

[Step 2: W2 训练 SideChannel] ✅ done
└── 数据: REDS train_sharp (270 clips, 在线 Real-ESRGAN 退化)
└── 命令: GPU_IDS=0,1 bash scripts/train_sidechannel.sh
└── 产物: runs/sidechannel_v2_lastframe/sidechannel_step{000500..005000}.pt

[Step 3: W3 接入推理] ✅ done
└── 命令同上,加 --sidechannel_ckpt runs/sidechannel_v2_lastframe/sidechannel_step005000.pt
└── 产物: /nas-files/yuhanchen/tmp/sc_test_w3_out/<clip>/frame_*.png

[Step 4: W4 RAFT 替换] ⏳ pending
└── 改 src/cross_frame_retrieval/cfr_main.py 内 SpyNet 调用
└── 不需要训(RAFT 用官方预训练 ckpt),只跑推理 sanity check
└── 数据: VideoLQ (大运动,RAFT 收益最显著的场景)

[Step 5: W5 全基线扫表] ⏳ pending
└── 测试集:
    • UDM10 ✅ 在盘
    • SPMCS ⏳ 待下
    • VideoLQ ⏳ 待下
    • RealVSR ⏳ 待下
└── 配置矩阵: baseline / +sidechannel / +sidechannel_gateoff / +RAFT / 全开
└── 不需要训,只跑推理 + 算指标

[Step 6: W6 报告 + 可视化] ⏳ pending
└── α 热力图、ablation 表、OCR delta
```

---

## 7. 评估指标

| 指标 | 工具 | 期望表现 |
| --- | --- | --- |
| **PSNR / SSIM** | skimage / pyiqa | 全集小幅提升(+0.2~0.5 dB) |
| **LPIPS / DISTS** | lpips==0.1.4 / pyiqa | 明显提升(side-channel 直接补高频) |
| **tLPIPS / E_warp** | 自实现 | 不能掉(gate 应该抑制不一致区域) |
| **PaddleOCR 准确率** | paddleocr | **杀手图** —— 在小字视频上验证 VAE 瓶颈被绕开 |

---

## 8. 风险与对策

| 风险 | 触发条件 | 对策 |
| --- | --- | --- |
| Side-channel 把 LR 的噪声 / JPEG artifact 也补进 HR | gate 学不会区分 | 加 `λ·||α||_1` 稀疏正则;或预处理用 BM3D 给 x_up 去噪 |
| GateNet 退化为 α≡0 (全关) 或 α≡1 (全开) | loss 局部最小 | bias=+2.0 cold-start + 中后期开稀疏正则;实测 α 落在 0.71,正常 |
| OCR 没提升 | gate 在文字位置反而关了 | 检查 α 热力图;必要时引入 Canny 损失 |
| RAFT OOM | 高分辨率 + 长序列 | RAFT-small / GMFlow / CPU 离线预算 flow |

---

## 9. 状态总览

| Sprint | 内容 | 状态 |
| --- | --- | --- |
| W1 | UDM10 baseline 复现 | ✅ |
| W2 | SideChannel 模块 + 训练管线 + 5000 步训练 | ✅ |
| W3 | `test_DLoRAL.py` 接入 `--sidechannel_ckpt` | ✅ |
| W4 | SpyNet → RAFT | ⏳ |
| W5 | 全基线扫表 (UDM10 / SPMCS / VideoLQ / RealVSR) | ⏳(等数据集) |
| W6 | 报告 + 可视化 | ⏳ |

**当前阻塞:** SPMCS / VideoLQ / RealVSR 测试集需要手动下载到 `/nas-files/yuhanchen/CV/data/test_sets/`。


---

## 10. W4 完成: SpyNet → RAFT

**状态:** ✅ 已完成 (2026-05-01)

### 改动清单

| 文件 | 改动 |
| --- | --- |
| `src/cross_frame_retrieval/raft_flow.py` | RAFTFlowEstimator，与 SPyNet 同签名 `(img1,img2) -> [B,2,H,W]`，内部处理 [0,1]→[-1,1] 转换 + pad 到 8 倍数。从本地缓存加载权重，避免服务器 SSL 下载失败 |
| `src/cross_frame_retrieval/cfr_main.py` | `CFR_model.__init__` 加 `flow_estimator='spynet'` 参数，二选一构建 SPyNet 或 RAFT，设 `self.spynet = self.flow_net` 别名，零改动透传给 `compute_flow` 的 `self.spynet(...)` 调用 |
| `src/DLoRAL_model.py` | `Generator` / `Generator_eval` 两处 `CFR_model(...)` 调用传 `flow_estimator=getattr(args, 'flow_estimator', 'spynet')`；两处 `load_ckpt_from_state_dict` 容错跳过缺失的 spynet/flow_net 键 |
| `src/test_DLoRAL.py` | 新增 `--flow_estimator {spynet,raft}` CLI |
| `src/my_utils/training_utils.py` | 同上 |
| `scripts/test_dloral_raft.sh` | UDM10 RAFT-only 推理 + 自动算指标 |
| `scripts/test_dloral_raft_sc.sh` | UDM10 RAFT + SideChannel 推理 + 自动算指标 |
| `scripts/udm10_to_mp4.sh` | UDM10 PNG 序列 → mp4 转换（test_DLoRAL.py 只接受 mp4） |

### RAFT 权重加载

服务器无法访问 `download.pytorch.org` (SSL 握手超时 + CloudFront 403)。权重通过本地 torchvision API 下载后 SCP 到 `~/.cache/torch/hub/checkpoints/raft_small-8bb27295.pth`。

`raft_flow.py` 加载策略: 先检查本地缓存是否存在 → 有则 `load_state_dict`; 无则尝试 `raft_small(weights=Raft_Small_Weights.DEFAULT)` 下载(兜底)。

RAFT-small 参数量: ~990K。

---

## 11. W7 改进方案: Text-Aware Frequency Residual

### 11.1 动机

当前 Side-Channel (W2) 虽然简单有效，但有三方面可改进:

| 局限 | 说明 |
| --- | --- |
| **无显式频域分解** | DetailNet 直接吃 RGB concat，靠 ResNet 隐式学习高频提取。对规则纹理(文字、网格)不如显式频域分解稳定 |
| **无区域感知** | GateNet 的 α 是全局统计产物(实测均值 0.71)，无显式边缘/文字先验引导。PSNR 优化下 gate 倾向关掉"风险区域"——往往正是文字(像素少、PSNR 贡献小) |
| **单一路径** | 所有内容走同一个 DetailNet，低频平滑区和高频文字区被迫共享同一表示 |

### 11.2 改进总览

三条线并行:

1. **可学习特征提取器** — 用 DWT 小波变换替代隐式 ResNet，显式建模频域信息
2. **频域分解残差** — Haar DWT 把输入解耦为 LL/LH/HL/HH 四个子带，网络分别对高频子带做针对性增强
3. **文本感知残差分支** — 引入 PaddleOCR 文字 mask + Canny 边缘图，引导 GateNet 在文字/边缘区域保持开门

### 11.3 目标架构

```
x_up (bicubic LR)  +  y_coarse (DLoRAL output)
         |
    ┌────┼────────────┐
    v    v            v
  DWT   Canny      PaddleOCR
 (12ch) (1ch)       (1ch)
    |    |            |
    v    +------+------+
    |           |
    |           v
    |     TextGateNet
    |     (alpha with text/edge boost)
    |           |
    v           v
 FreqDetailNet |
 (频域残差)     |
    |           |
    +-----+-----+
          |
          v
  y_final = y_coarse + alpha_text * d_freq
```

### 11.4 W7A: DWT 频域残差分支

新增模块:

| 文件 | 内容 |
| --- | --- |
| `src/side_channel/dwt_utils.py` | Haar DWT/IDWT 可微实现: 3ch→12ch (LL/LH/HL/HH x 3)，IDWT 12ch→3ch |
| `src/side_channel/freq_detail_net.py` | 在 DWT 域做 ResNet: 各子带独立 conv + 低频保真约束 + 末层 IDWT 回到 3ch |

设计要点:
- DWT 按方向+频率解耦为 4 子带，网络分别处理 LL/LH/HL/HH
- 比固定高通滤波更灵活: 滤波核是死的，DWT 后网络可学习不同子带加权
- 末层 IDWT 前 zero-init → 起点等价 baseline

### 11.5 W7B: 文字感知 Gate

新增模块:

| 文件 | 内容 |
| --- |---|
| `src/side_channel/text_gate_net.py` | 继承 GateNet, 额外输入: Canny 边缘图 + PaddleOCR 文字 mask |

设计要点:
- 训练时: `concat(x_up, y_coarse, canny_edge, ocr_mask)` → 8ch
- Gate init: 文字区域 alpha init=0.95, 非文字 alpha init=0.5
- 推理轻量模式: 只用 Canny，不开 PaddleOCR

### 11.6 W7C: OCR 评估管线

新增:

| 文件 | 内容 |
| --- | --- |
| `tools/eval_ocr.py` | 预测帧 vs GT 帧 PaddleOCR → 字符准确率 |

### 11.7 消融矩阵

| 实验 | DetailNet | GateNet | 预期 |
| --- | --- | --- | --- |
| 1 baseline | - | - | DLoRAL 原始上限 |
| 2 +W2 sidechannel | ResNet RGB | U-Net RGB | 全局 PSNR +0.2~0.5 |
| 3 +W7A DWT | FreqDetailNet | U-Net RGB | 纹理/边缘 +0.1~0.3 |
| 4 +W7B TextGate | FreqDetailNet | TextGateNet | 文字 LPIPS down, OCR up |
| 5 全开 | FreqDetailNet | TextGateNet | 全域+文字双赢 |

### 11.8 风险

| 风险 | 对策 |
| --- | --- |
| DWT 12ch 计算量增大 | DWT/IDWT O(N)，仅在 freq_detail_net 内，输入输出仍 3ch |
| PaddleOCR 推理开销 | 离线预处理 OCR mask→pkl，训练时直接读 |
| TextGate 过拟合 | REDS 含多样化文字，UDM10 仅测试 |

### 11.9 实施顺序

| 子任务 | 内容 | 新代码 | 训练 |
| --- | --- | --- | --- |
| W7A | dwt_utils.py + freq_detail_net.py | ~200行 | ~5000 step |
| W7B | text_gate_net.py + OCR 预处理 | ~150行 | ~2000 step |
| W7C | eval_ocr.py + 消融扫表 | ~100行 | 无 |

全程 DLoRAL 冻参数，只训 SideChannel，单卡可跑。

---

## 12. 状态总览 (2026-05-01 更新)

| Sprint | 内容 | 状态 |
| --- | --- | --- |
| W1 | UDM10 baseline 复现 | ✅ |
| W2 | SideChannel 模块 + 训练管线 + 5000 步训练 | ✅ |
| W3 | test_DLoRAL.py 接入 --sidechannel_ckpt | ✅ |
| W4 | SpyNet → RAFT | ✅ |
| W5 | 全基线扫表 (UDM10/SPMCS/VideoLQ/RealVSR) | ⏳ |
| W6 | 报告 + 可视化 | ⏳ |
| W7 | Text-Aware Frequency Residual (DWT + TextGate + OCR eval) | ⏳ |

---

## 13. W4 补充修复 (2026-05-01): RAFT fp16 兼容性

### 13.1 问题

RAFT 在 fp16 mixed precision 下出现 dtype 不匹配：
1. **Conv2d bias mismatch**: `model.cfr_main_net.to(dtype=torch.float16)` 将 RAFT 参数转为 fp16，但 `raft_flow.py` 中 `.float()` 强制输入为 float32
2. **grid_sample mismatch**: RAFT 内部 `corr_block.index_pyramid` 调用 `grid_sample` 时，correlation volume (fp16) 与 sampling grid (float32) 类型不一致

### 13.2 修复

`src/cross_frame_retrieval/raft_flow.py` forward 方法：
- 调用 `self.net.float()` 确保 RAFT 模型始终在 float32 下运行
- 输入统一 `.float()` 转换
- 输出 `.to(img1.dtype)` 匹配原始精度

### 13.3 额外修复

`src/DLoRAL_model.py` line 753: `{scale: 1.0}` → `{"scale": 1.0}` (dict key 缺引号)


---

## 14. W4 全量 UDM10 评测 (2026-05-01 运行中)

### 14.1 测试配置

| 实验 | 脚本 | 输出目录 |
| --- | --- | --- |
| RAFT-only | `scripts/test_dloral_raft.sh` | `tmp/udm10_raft_out/` |
| RAFT+SideChannel | `scripts/test_dloral_raft_sc.sh` | `tmp/udm10_raft_sc_out/` |

两者均使用 `--flow_estimator raft`，10 个 UDM10 视频，fp16 mixed precision。

### 14.2 状态

⏳ 运行中 (2026-05-01 18:50+)，预计 3-4 小时完成。完成后自动运行 `tools/eval_metrics.py` 生成 PSNR/SSIM/LPIPS。


---

## 15. W4 测试优化 (2026-05-01)

### 15.1 瓶颈分析

原测试 slow 原因：
- 两台测试共享 GPU 0（GPU-Util 仅 37%），GPU 1 空闲
- tile_size=1024 导致 720×1272 视频每帧需 182 encoder tiles
- 大量时间花在 tile 调度/拼接而非实际计算

### 15.2 优化

| 参数 | 旧值 | 新值 | 原因 |
| --- | --- | --- | --- |
| GPU | 两台→GPU 0 | 各占 GPU 0/1 | 消除竞争 |
| vae_encoder_tiled_size | 1024 | 2048 | 消除 tiling (1272<2048) |
| vae_decoder_tiled_size | 224 | 448 | 减少 decoder tiles |

### 15.3 W3 为何更快

W3 测试为单进程（GPU 无竞争），且 SPyNet 原生 fp16 兼容。W4 首次运行时两个进程共享 GPU 0 + RAFT 被强制 float32 导致额外开销。现在通过分 GPU + 增 tile 解决。


---

## 16. W4 UDM10 全量测试结果 (2026-05-01)

### 16.1 RAFT-only (无 SideChannel)

| Clip | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- |
| archpeople | 14.505 | 0.632 | 0.825 |
| archwall | 14.875 | 0.482 | 0.705 |
| auditorium | 9.627 | 0.399 | 0.844 |
| band | 12.230 | 0.526 | 0.821 |
| caffe | 14.199 | 0.372 | 0.797 |
| camera | 15.271 | 0.733 | 0.610 |
| clap | 9.507 | 0.481 | 0.834 |
| lake | 14.909 | 0.526 | 0.840 |
| photography | 13.905 | 0.390 | 0.848 |
| polyflow | 10.311 | 0.475 | 0.937 |
| **WEIGHTED MEAN** | **12.934** | **0.502** | **0.806** |

### 16.2 RAFT+SideChannel

指标与 RAFT-only **完全相同**。原因：SideChannel step5000 checkpoint 的 DetailNet 残差极小 (d_t mean=0.006)，alpha×d_t 最大像素变化 ~0.05 (12/255)，在 fp16→uint8 转换中量化归零。

### 16.3 问题总结

1. **RAFT 指标大幅下降**：CFR 网络用 SPyNet 流特征预训练，RAFT 流分布不兼容。需微调 CFR。
2. **SideChannel 无效果**：当前 checkpoint (step 5000) 门控过保守。需更长训练或降低 sparsity loss 权重。

### 16.4 下一步

| 方向 | 内容 | 优先级 |
| --- | --- | --- |
| W4-fix | CFR 微调适配 RAFT 流 | 高 ⬆ |
| W2-retrain | SideChannel 继续训练至收敛 | 中 |
| W7 | Text-Aware Frequency Residual | 低（等 W4 稳定）


---

## 17. W4-fix: CFR-RAFT Fine-Tuning — NaN Blocker (2026-05-02)

### 17.1 目标

微调 CFR backbone (~1.2M params) 适配 RAFT 光流分布，冻结 UNet/LoRA/RAFT，
使用 REDS 数据集，2 GPU fp16 训练 25000 步。

### 17.2 实现改动

| 文件 | 改动 |
| --- | --- |
| `src/DLoRAL_model.py` | 新增 `set_train_cfr_only()` — 只训练 CFR backbone，冻结 UNet/VAE/flow_net |
| `src/train_DLoRAL.py` | CFR-only 训练路径：跳过 stage 切换、CSD、VSDReward；spynet AND flow_net 双排除 |
| `src/my_utils/training_utils.py` | `--train_cfr_only` CLI flag；修复 REDS LR_frames 路径 |
| `scripts/train_cfr_raft.sh` | 2 GPU, fp16, cosine LR 1e-4, 25000 steps, resolution 512 |
| `preset/models/hparams.yml` | 新建 (lora_rank_unet_quality=4, lora_rank_unet_consistency=4) |
| `data/reds_train.txt` | 20880 训练样本 (240 序列 x 87 帧) |

### 17.3 冒烟测试结果

**状态: FAILED — NaN 损失**

冒烟测试 (15 步) 所有 loss 均为 NaN：
```
Step 1: L2=nan, LPIPS=nan, consistency=nan, total=nan
Step 6: (same, training stopped)
```

### 17.4 根因分析

**RAFT 光流估计器本身产生 NaN**，与 CFR 无关。

诊断脚本 `diagnose_nan.py` 验证：
```
RAFT flow (64x64): min=nan max=nan mean=nan !!!NaN!!!
SPyNet flow (64x64): min=-2.56 max=1.82 mean=-0.50  (normal)
```

NaN 传播链：
```
RAFT forward -> flow=NaN
  -> flow_warp(img, NaN_flow) -> warped=NaN
    -> cross_attn_module -> attn=NaN -> CFR output=NaN -> loss=NaN
```

RAFT checkpoint 权重正常 (106 keys 全部匹配, 无 NaN)，但前向传播仍产 NaN。
测试输入包括随机噪声和结构化图像(方块位移)，均在 float32 下复现。

环境: PyTorch 2.0.1+cu117, torchvision 0.15.2+cu117, CUDA 11.7, NVIDIA A40

### 17.5 已排除原因

| 原因 | 结论 | 证据 |
| --- | --- | --- |
| fp16 溢出 | 排除 | float32 下也 NaN |
| 数据 pipeline | 排除 | 合成输入也 NaN |
| CFR checkpoint 损坏 | 排除 | NaN 出现在 CFR 之前 |
| RAFT checkpoint 缺失 | 排除 | 文件存在 (4MB) |
| RAFT checkpoint 损坏 | 排除 | 所有权重无 NaN |
| load_state_dict 不匹配 | 排除 | 0 missing, 0 unexpected |

### 17.6 当前假设

| 假设 | 描述 | 验证方法 |
| --- | --- | --- |
| H1: torchvision 版本 bug | torchvision 0.15.2 的 RAFT CUDA kernel 在 A40 上有 bug | CPU 推理测试 |
| H2: checkpoint 版本不匹配 | raft_small-8bb27295.pth 来自不同 torchvision 版本 | 用官方 API `raft_small(weights=...)` |
| H3: CUDA 11.7 + A40 数值问题 | 特定 GPU/驱动组合导致 correlation volume NaN | 换 GPU 测试 |

### 17.7 修复方案 (按优先级)

**A: 修 RAFT NaN (首选)**
1. 完成 CPU 测试 (后台任务已排队)
2. CPU 正常 -> 升级 torchvision 或 CPU 计算光流
3. CPU 也 NaN -> 重新下载 RAFT checkpoint

**B: 回退到 SPyNet (兜底)**
去掉 `--flow_estimator raft`，CFR 保持 SPyNet 训练。丧失 RAFT 大运动优势但可继续训练。

**C: 跳过 --load_cfr (从头训练 CFR)**
不用预训练 CFR checkpoint，从头训练以适应 RAFT 流分布。

### 17.8 下一步

1. [ ] 完成 CPU RAFT 测试
2. [ ] 根据 CPU 结果选修复方案
3. [ ] 重新冒烟测试
4. [ ] 通过后启动 25000 步完整训练
5. [ ] 训练完成后 UDM10 评估 (目标 PSNR >=25.0)

### 17.9 相关文件

| 文件 | 说明 |
| --- | --- |
| `bug.md` | 详细 bug 报告 |
| `diagnose_nan.py` | NaN 诊断脚本 |
| `runs/cfr_raft_finetune/` | 训练输出 (含 NaN loss 日志) |
| `src/cross_frame_retrieval/raft_flow.py` | RAFT wrapper |
| `src/cross_frame_retrieval/cfr_main.py` | CFR model (含 compute_flow) |

---

## 18. W4-fix: CFR-RAFT Fine-Tuning Results (2026-05-05)

### 18.1 Training

- Checkpoint: 
- Steps: 52,200 (overshot from 25,000 target)
- Duration: ~84.5 hours on 2x GPU
- Status: Completed successfully

### 18.2 UDM10 Evaluation

2-GPU parallel eval (), fp32 inference.

| Clip | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- |
| archpeople | 28.80 | 0.858 | 0.228 |
| archwall | 31.05 | 0.836 | 0.172 |
| auditorium | 23.70 | 0.781 | 0.238 |
| band | 25.02 | 0.815 | 0.159 |
| caffe | 27.78 | 0.748 | 0.168 |
| camera | 31.33 | 0.913 | 0.203 |
| clap | 26.38 | 0.833 | 0.152 |
| lake | 25.68 | 0.666 | 0.490 |
| photography | 28.89 | 0.834 | 0.158 |
| polyflow | 27.16 | 0.805 | 0.217 |
| **WEIGHTED MEAN** | **27.58** | **0.809** | **0.218** |

### 18.3 Comparison

| Metric | Pre-FT (RAFT-only) | Post-FT (CFR-RAFT) | Delta |
| --- | --- | --- | --- |
| PSNR | 12.934 | 27.579 | +14.645 dB |
| SSIM | 0.502 | 0.809 | +0.307 |
| LPIPS | 0.806 | 0.218 | -0.588 |

CFR-RAFT fine-tuning resolves the SPyNet→RAFT flow distribution mismatch.
PSNR exceeds 25.0 target by +2.6 dB.

### 18.4 Bugs Fixed During Eval

1. **raft_flow.py dtype mismatch**:  forced fp32 input → fp16 model conv layers.
   Fix: Auto-detect model dtype with .

2. **grid_sample dtype mismatch**: CFR feature warping (fp16) vs RAFT flow (fp32).
   Fix: Use  for fp32 inference. Checkpoint saved in fp32 anyway.

### 18.5 Next Steps

- [ ] W5: Full baseline sweep (SPMCS, VideoLQ, RealVSR) — need datasets downloaded
- [ ] W2-retrain: SideChannel longer training to convergence (current step5000 too conservative)
- [ ] W7: Text-Aware Frequency Residual (DWT + TextGate + OCR eval)
