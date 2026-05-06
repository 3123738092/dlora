<div align="center">
<h2>One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution</h2>

[Yujing Sun](https://yjsunnn.github.io/)<sup>1,2, *</sup> | 
[Lingchen Sun](https://scholar.google.com/citations?hl=zh-CN&tzom=-480&user=ZCDjTn8AAAAJ)<sup>1,2, *</sup> | 
[Shuaizheng Liu](https://scholar.google.com/citations?user=wzdCc-QAAAAJ&hl=en)<sup>1,2</sup> | 
[Rongyuan Wu](https://scholar.google.com/citations?user=A-U8zE8AAAAJ&hl=zh-CN)<sup>1,2</sup> | 
[Zhengqiang Zhang](https://scholar.google.com.tw/citations?user=UX26wSMAAAAJ&hl=en)<sup>1,2</sup> | 
[Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)<sup>1,2</sup>

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute

<h3>📍  NeurIPS 2025</h3>

</div>

<div>
    <h4 align="center">
        <a href="https://yjsunnn.github.io/DLoRAL-project/" target='_blank'>
        <img src="https://img.shields.io/badge/💡-Project%20Page-gold">
        </a>
        <a href="https://arxiv.org/pdf/2506.15591" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2312.06640-b31b1b.svg">
        </a>
        <a href="https://www.youtube.com/embed/Jsk8zSE3U-w?si=jz1Isdzxt_NqqDFL&vq=hd1080" target='_blank'>
        <img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white">
        </a>
        <a href="https://www.youtube.com/embed/xzZL8X10_KU?si=vOB3chIa7Zo0l54v" target="_blank">
        <img src="https://img.shields.io/badge/2--Min%20Explainer-brightgreen?logo=YouTube&logoColor=white">
        </a>
        </a>
        <a href="https://zhuanlan.zhihu.com/p/1959430260706744130" target="_blank">
        <img src="https://img.shields.io/badge/Zhihu-0084FF?style=flat&logo=zhihu&logoColor=white">
        </a>
        </a>
        <a href="https://github.com/yjsunnn/Awesome-video-super-resolution-diffusion" target="_blank">
        <img src="https://img.shields.io/badge/GitHub-Awesome--VSR--Diffusion-181717.svg?logo=github&logoColor=white">
        </a>
        </a>
        <a href="https://colab.research.google.com/drive/1QAEn4uFe4GNqlJbogxxhdGFhzMr3rfGm?usp=sharing" target="_blank">
        <img src="https://img.shields.io/badge/Colab%20Demo-F9AB00?style=flat&logo=googlecolab&logoColor=white">
        </a>
        <a href="https://github.com/yjsunnn/DLoRAL" target='_blank' style="text-decoration: none;"><img src="https://visitor-badge.laobi.icu/badge?page_id=yjsunnn/DLoRAL"></a>
    </h4>
</div>

<p align="center">

<img src="assets/visual_results.svg" alt="Visual Results">

</p>

## ⏰ Update

- **2025.10.16**: We update [**an improved version**](https://drive.google.com/file/d/1C2TLERta3a-PkoMpqQhHoM_S54pNEASO/view?usp=drive_link) of DLoRAL. Thanks [@Feynman1999](https://github.com/Feynman1999) for the bug fixes!
- **2025.09.18**: DLoRAL is accepted by **NIPS2025** 🎉
- **2025.07.14**: [**Colab demo**](https://colab.research.google.com/drive/1QAEn4uFe4GNqlJbogxxhdGFhzMr3rfGm?usp=sharing) is available. ✨ **No local GPU or setup needed** - just upload and enhance!
- **2025.07.08**: The inference code and pretrained weights are available.
- **2025.06.24**: The project page is available, including a brief 2-minute explanation video, more visual results and relevant researches.
- **2025.06.17**: The repo is released.

:star: If DLoRAL is helpful to your videos or projects, please help star this repo. Thanks! :hugs:

😊 You may also want to check our relevant works:

1. **OSEDiff (NIPS2024)** [Paper](https://arxiv.org/abs/2406.08177) | [Code](https://github.com/cswry/OSEDiff/)  

   Real-time Image SR algorithm that has been applied to the OPPO Find X8 series.

2. **PiSA-SR (CVPR2025)** [Paper](https://arxiv.org/pdf/2412.03017) | [Code](https://github.com/csslc/PiSA-SR) 

   Pioneering exploration of Dual-LoRA paradigm in Image SR.
3. **TVT-SR (ICCV2025)** [Paper](https://arxiv.org/pdf/2507.20291) | [Code](https://github.com/Joyies/TVT)
   
   A compact VAE and compute-efficient UNet able to handle fine-grained structures.

5. **Awesome Diffusion Models for Video Super-Resolution** [Repo](https://github.com/yjsunnn/Awesome-video-super-resolution-diffusion)

   A curated list of resources for Video Super-Resolution (VSR) using Diffusion Models.

## 👀 TODO
- [x] Release inference code.
- [x] Colab demo for convenient test.
- [x] Release training code.
- [ ] Release training data.


## 🌟 Overview Framework

<p align="center">

<img src="assets/pipeline.svg" alt="DLoRAL Framework">

</p>

**Training**: A dynamic dual-stage training scheme alternates between optimizing temporal coherence (consistency stage) and refining high-frequency spatial details (enhancement stage) with smooth loss interpolation to ensure stability.

**Inference**: During inference, both C-LoRA and D-LoRA are merged into the frozen diffusion UNet, enabling one-step enhancement of low-quality inputs into high-quality outputs.


## 🔧 Dependencies and Installation

1. Clone repo
    ```bash
    git clone https://github.com/yjsunnn/DLoRAL.git
    cd DLoRAL
    ```

2. Install dependent packages
    ```bash
    conda create -n DLoRAL python=3.10 -y
    conda activate DLoRAL
    pip install -r requirements.txt
    # mim install mmedit and mmcv
    pip install openmim
    mim install mmcv-full
    pip install mmedit
    ```

3. Download Models 
#### Dependent Models
* [RAM](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth) --> put into **/path/to/DLoRAL/preset/models/ram_swin_large_14m.pth**
* [DAPE](https://drive.google.com/file/d/1KIV6VewwO2eDC9g4Gcvgm-a0LDI7Lmwm/view?usp=drive_link) --> put into **/path/to/DLoRAL/preset/models/DAPE.pth**
* [Pretrained Weights](https://drive.google.com/file/d/1C2TLERta3a-PkoMpqQhHoM_S54pNEASO/view?usp=drive_link) --> put into **/path/to/DLoRAL/preset/models/checkpoints/model.pkl**
  - If your goal is to reproduce the results from the paper, we recommend using this version of the [weights](https://drive.google.com/file/d/1ycreq0wxmqVUaN3ORQXJfCtqDyD7RlUV/view?usp=drive_link) instead.


Each path can be modified according to its own requirements, and the corresponding changes should also be applied to the command line and the code.

## 🖼️ Quick Inference
For Real-World Video Super-Resolution:

```
python src/test_DLoRAL.py     \
--pretrained_model_path yujingsun/stable-diffusion-2-1-base     \
--ram_ft_path /path/to/DLoRAL/preset/models/DAPE.pth     \
--ram_path '/path/to/DLoRAL/preset/models/ram_swin_large_14m.pth'     \
--merge_and_unload_lora False     \
--process_size 512     \
--pretrained_model_name_or_path yujingsun/stable-diffusion-2-1-base     \
--vae_encoder_tiled_size 4096     \
--load_cfr     \
--pretrained_path /path/to/DLoRAL/preset/models/checkpoints/model.pkl     \
--stages 1     \
-i /path/to/input_videos/     \
-o /path/to/results
```

## ⚙️ Training
For Real-World Video Super-Resolution:

```
bash train_scripts.sh
```

Some key parameters and corresponding meaning:
Param | Description | Example Value
--- | --- | ---
`--quality_iter` | Number of steps for the initial switch from consistency to quality stage | `5000`
`--quality_iter_1_final` | Number of steps required to switch from the quality stage to the consistency stage | `13000`
`--quality_iter_2` | Relative number of steps after `quality_iter_1_final` to switch back to the quality stage (actual switch happens at `quality_iter_1_final + quality_iter_2`) | `5000`
`--lsdir_txt_path` | Dataset path for the first stage | `"/path/to/your/dataset"`
`--pexel_txt_path` | Dataset path for the second stage | `"/path/to/your/dataset"`



## 💬 Contact:
If you have any problem (not only about DLoRAL, but also problems regarding to burst/video super-resolution), please feel free to contact me at yujingsun1999@gmail.com

### Citations
If our code helps your research or work, please consider citing our paper.
The following are BibTeX references:

```
@misc{sun2025onestepdiffusiondetailrichtemporally,
      title={One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution}, 
      author={Yujing Sun and Lingchen Sun and Shuaizheng Liu and Rongyuan Wu and Zhengqiang Zhang and Lei Zhang},
      year={2025},
      eprint={2506.15591},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.15591}, 
}

---

## Side-Channel Extension (CV course P1)

> Custom extension on top of upstream DLoRAL. Adds a pixel-space residual head
> bypassing the lossy 8× VAE to recover high-frequency detail. Upstream code is
> unchanged; new modules live under `src/side_channel/`.

### What was added

| File | Purpose |
| --- | --- |
| `src/side_channel/__init__.py` | Package entry — exports `DetailNet`, `GateNet`, `SideChannelWrapper`. |
| `src/side_channel/detail_net.py` | ResNet-style residual predictor (`d_t`, ~4.88M params, zero-init tail). |
| `src/side_channel/gate_net.py` | Symmetric U-Net producing per-pixel α ∈ [0,1] (~1.88M params, sigmoid bias init = 2.0). |
| `src/side_channel/side_channel_wrapper.py` | Combines them as `y_final = y_coarse + α · d_t`. |
| `src/side_channel/dataset.py` | REDS multi-frame loader with on-the-fly Real-ESRGAN degradation. |
| `src/train_side_channel.py` | Accelerate-based DDP training loop (frozen DLoRAL → side-channel). |
| `configs/side_channel.yaml` | Training config (5000 steps, lr=1e-4, λ_lpips=0.5, hr_size=512, num_frames=2). |
| `configs/side_channel_dryrun.yaml` | 2-step smoke config for plumbing checks. |
| `scripts/train_sidechannel.sh` | Multi-GPU launcher (`accelerate launch --num_processes=N`). |
| `requirements.txt` | Added `mmengine==0.10.7`, `mmcv==2.1.0` (cu117/torch2.0 wheel), `lpips==0.1.4`. |

### Architecture

```
LR ──Real-ESRGAN───────► HR_target          (training only)
       │
       └─bicubic-up──► x_up ──┐
                              ├─► frozen DLoRAL ──► y_coarse  (8× VAE bottleneck)
   neighbor frame ────────────┘
                              │
                              ▼
              SideChannelWrapper(x_up, y_coarse)
                  ├─ DetailNet → d_t   (~4.88M, residual)
                  └─ GateNet   → α     (~1.88M, gating)
                              │
                              ▼
                 y_final = y_coarse + α · d_t
```

Zero-init guarantees `y_final == y_coarse` at step 0 — never degrades the baseline.
Window last frame (`[:, -1]`) is supervised, matching DLoRAL's output convention
(`src_idx = input_image_index + start + 1`).

### Reproducing the training run

Prereq: REDS train clips at `/data/yuhanchen/CV/data/reds/train/train_sharp/{000..269}/*.png`,
DLoRAL checkpoint at `preset/models/checkpoints/model.pkl`,
SD-2.1 base at `preset_models/stable-diffusion-2-1-base/`,
`params.yml` at `src/datasets/params.yml` (Real-ESRGAN degradation settings, included).

Smoke test (2 steps, single GPU):

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_side_channel.py \
    --config configs/side_channel_dryrun.yaml
```

Production training (2 GPUs, ~80 min wallclock):

```bash
GPU_IDS=0,1 bash scripts/train_sidechannel.sh
# equivalent to:
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
#   --num_processes=2 --num_machines=1 --mixed_precision=no \
#   src/train_side_channel.py --config configs/side_channel.yaml
```

Checkpoints land in `runs/sidechannel_v2_lastframe/sidechannel_step{000500..005000}.pt`
(saved every 500 steps).

### Datasets used

* **Train**: REDS `train_sharp` (270 clips × 100 frames, 23,760 sliding-window samples
  with `num_frames=2`) — already present at `/data/yuhanchen/CV/data/reds/`.
* **Eval (UDM10)**: reproduced via VRT release tarball; converted BDx4 PNGs to mp4 with
  `ffmpeg -framerate 10 -i frame_%08d.png -c:v libx264 -pix_fmt yuv420p -crf 12`.
* **Eval (REDS4)**: subset of REDS val (`val_sharp/{000,011,015,020}`) — already on disk.
* **Eval (SPMCS / VideoLQ / RealVSR)**: pending manual download by user; place under
  `/nas-files/yuhanchen/CV/data/test_sets/`.

### Inference with side-channel (W3, done)

Two new CLI flags on `src/test_DLoRAL.py` (default off → identical to upstream):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--sidechannel_ckpt PATH` | `None` | Path to a `sidechannel_step*.pt`. Disabled when omitted. |
| `--sidechannel_alpha_scale FLOAT` | `1.0` | Multiplier on gate α at inference. `0.0` ≡ disabled, `1.0` ≡ as trained, `>1` exaggerates the residual (debug only). |

Behaviour: after the frozen DLoRAL produces `output_image` (window last frame), the
wrapper computes `y_final = y_coarse + α · d_t` in `[0,1]` space, then re-maps to
`[-1,1]` for the existing color-fix / save path. `output_image[0]` and the
`adain`/`wavelet` color correction downstream see the corrected tensor without
further changes.

Reference command (UDM10, GPU 0):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
/nas-files/yuhanchen/miniconda3/envs/lora/bin/python src/test_DLoRAL.py \
    --pretrained_model_path preset_models/stable-diffusion-2-1-base \
    --pretrained_model_name_or_path preset_models/stable-diffusion-2-1-base \
    --ram_ft_path preset/models/DAPE.pth \
    --ram_path  preset/models/ram_swin_large_14m.pth \
    --merge_and_unload_lora False \
    --process_size 512 --vae_encoder_tiled_size 4096 \
    --load_cfr --pretrained_path preset/models/checkpoints/model.pkl \
    --stages 1 --align_method adain \
    --sidechannel_ckpt runs/sidechannel_v2_lastframe/sidechannel_step005000.pt \
    -i /nas-files/yuhanchen/tmp/sc_test_w3_in \
    -o /nas-files/yuhanchen/tmp/sc_test_w3_out
```

Notes:
* `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` keep `bert-base-uncased` and SD weights
  resolution local; default `~/.cache/huggingface/hub` already has the snapshots.
* Patch backup at `src/test_DLoRAL.py.bak_w3` (use `diff` to inspect the four
  insertion points: import / argparse / wrapper init / inference hook).

### Roadmap

| Sprint | Status | Deliverable |
| --- | --- | --- |
| W1 | ✅ done | UDM10 baseline reproduced (frame-by-frame DLoRAL output saved). |
| W2 | ✅ done | `src/side_channel/` package + 5000-step DDP training run; loss `l1 0.30→0.06`, `lpips 0.70→0.32`, gate α settled around 0.71. |
| W3 | ✅ done | `--sidechannel_ckpt` flag; pixel-space residual hooked into inference tail. |
| W4 | ⏳ pending | SpyNet → RAFT inside `src/cross_frame_retrieval/cfr_main.py`; expected ~0.5–1 dB PSNR on large-motion clips. |
| W5 | ⏳ pending | Full benchmark sweep (UDM10 / SPMCS / VideoLQ / RealVSR) + ablations: α=0 (baseline), α=1 (trained), gate-disabled (α≡1). |
| W6 | ⏳ pending | Course report + visualisations (gate heatmaps, OCR delta on small-text clips). |

Pending data-side work: SPMCS / VideoLQ / RealVSR test sets to be downloaded into
`/nas-files/yuhanchen/CV/data/test_sets/` before W5.

