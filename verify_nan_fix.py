#!/usr/bin/env python3
"""Verify RAFT NaN fix: upsampling small inputs to >=128 before RAFT."""
import os, sys
os.environ['TRANSFORMERS_OFFLINE'] = '0'
sys.path.insert(0, '/nas-files/yuhanchen/lora/DLoRAL/src')

import torch
import torch.nn.functional as F
from src.cross_frame_retrieval.raft_flow import RAFTFlowEstimator
from src.cross_frame_retrieval.cfr_main import CFR_model

def stats(name, t):
    x = t.detach().float()
    nan = torch.isnan(x).any().item()
    return "{}: shape={} min={:.6f} max={:.6f} {}".format(
        name, list(t.shape), x.min(), x.max(),
        "!!!NaN!!!" if nan else "OK")

print("=== RAFT NaN Fix Verification ===\n")

# Test 1: Direct RAFT at 64x64 (the failing case)
print("--- Test 1: RAFT at 64x64 ---")
raft = RAFTFlowEstimator(pretrained=True, small=True).cuda().eval()

img1 = torch.rand(1, 3, 64, 64).cuda()
img2 = torch.rand(1, 3, 64, 64).cuda()
img1[:, :, 10:50, 10:50] = 0.8
img2[:, :, 15:55, 15:55] = 0.8

with torch.no_grad():
    flow = raft(img1, img2)
print(stats("RAFT 64x64", flow))

# Test 2: RAFT at 128x128 (should still work)
print("--- Test 2: RAFT at 128x128 ---")
img1b = F.interpolate(img1, size=(128, 128), mode='bilinear')
img2b = F.interpolate(img2, size=(128, 128), mode='bilinear')
with torch.no_grad():
    flow128 = raft(img1b, img2b)
print(stats("RAFT 128x128", flow128))

# Test 3: RAFT at 32x32 (smaller than before)
print("--- Test 3: RAFT at 32x32 ---")
img1c = F.interpolate(img1, size=(32, 32), mode='bilinear')
img2c = F.interpolate(img2, size=(32, 32), mode='bilinear')
with torch.no_grad():
    flow32 = raft(img1c, img2c)
print(stats("RAFT 32x32", flow32))

# Test 4: CFR full forward with RAFT flow
print("\n--- Test 4: CFR forward with RAFT (512 LQ, 64 downsample) ---")
cfr = CFR_model(
    mid_channels=64, num_blocks=7, is_low_res_input=False,
    spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth',
    flow_estimator='raft')
cfr = cfr.cuda().eval()

ckpt_path = '/nas-files/yuhanchen/lora/DLoRAL/preset/models/checkpoints/model.pkl'
if os.path.exists(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cuda')
    cfr_sd = {}
    for k, v in sd.items():
        if 'cfr_main_net' in k:
            cfr_sd[k.replace('cfr_main_net.', '')] = v
    if cfr_sd:
        missing, unexpected = cfr.load_state_dict(cfr_sd, strict=False)
        print("CFR ckpt: {} params loaded (missing={}, unexp={})".format(
            len(cfr_sd), len(missing), len(unexpected)))

B, T = 1, 2
H, W = 512, 512
Hl, Wl = 64, 64
torch.manual_seed(42)
lqs_test = torch.rand(B, T, 3, H, W).cuda()
feat_test = torch.randn(B, T, 4, Hl, Wl).cuda() * 0.5
uncert_test = torch.rand(B, T, 1, Hl, Wl).cuda()

with torch.no_grad():
    fused, wmap, aligned = cfr(lqs_test, feat_test, uncert_test)
print(stats("CFR fused_features", fused))
print(stats("CFR weight_map", wmap))
print(stats("CFR aligned_feat", aligned))

has_nan = torch.isnan(fused).any().item()
print("\n" + ("*** PASS: No NaN detected ***" if not has_nan else "*** FAIL: NaN still present ***"))
