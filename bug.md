# BUG: RAFT flow estimator produces NaN at sub-128px input resolution

**Status:** FIXED (2026-05-02)
**Severity:** CRITICAL — was blocking CFR-only RAFT fine-tuning training
**Discovered:** 2026-05-01, during CFR+RAFT training smoke test
**Fixed:** 2026-05-02

---

## 1. Symptom

CFR-only fine-tuning produces NaN loss at every training step:

```
Step 1: loss=nan (L2=nan, LPIPS=nan, consistency=nan)
Step 2: loss=nan
...
Step 6: loss=nan  ← training was stopped here
```

## 2. Root Cause (CONFIRMED)

**RAFT-small's correlation pyramid produces NaN when input resolution < 128 pixels.**

CFR feeds flow estimators with 64x64 inputs after 8x bicubic downsample of the
512x512 LR frames (`cfr_main.py:645`):

```python
lqs_downsample = F.interpolate(lqs, scale_factor=0.125, mode='bicubic')  # 512->64
flows_forward, flows_backward = self.compute_flow(lqs_downsample)  # 64x64 -> RAFT
```

SPyNet handles 64x64 fine (pyramid-based architecture). RAFT-small does NOT
(correlation pyramid needs >=128px per dimension).

Empirical evidence:
| Input size | SPyNet | RAFT-small |
|------------|--------|------------|
| 32x32 | OK | NaN |
| 64x64 | OK (-2.56~1.82) | NaN |
| 96x96 | OK | NaN |
| 128x128 | OK | OK (-24.2~12.3) |

H1/H2/H3 (torchvision bug, checkpoint mismatch, CUDA+GPU issue) were all ruled out.
Same checkpoint + same torchvision 0.15.2 + same A40 GPU, 128x128 works perfectly.

## 3. NaN Propagation Chain

```
CFR: lqs (512x512) -> 8x downsample -> lqs_downsample (64x64)
  -> self.spynet(lqs_downsample)  [alias for RAFTFlowEstimator]
    -> RAFT.forward(64x64 input)
      -> correlation pyramid at 64x64 -> NaN flow
        -> flow_warp(img, NaN_flow) -> warped = NaN
          -> cross_attn_module -> attn = NaN
            -> CFR output = NaN -> loss = NaN
```

## 4. Fix Applied

**File:** `src/cross_frame_retrieval/raft_flow.py`

**Change:** In `RAFTFlowEstimator.forward()`, when input H or W < 128:
1. Bilinear upsample both images to >=128 (padded to multiple of 8)
2. Run RAFT on the upsampled images
3. Bilinear downsample the flow back to original resolution
4. Scale flow magnitudes proportionally (flow_x *= orig_w / upsampled_w, etc.)

```python
_MIN_SIZE = 128

if h < _MIN_SIZE or w < _MIN_SIZE:
    scale = max(_MIN_SIZE / h, _MIN_SIZE / w)
    new_h = int((h * scale + 7) // 8 * 8)
    new_w = int((w * scale + 7) // 8 * 8)
    x1u = F.interpolate(x1, size=(new_h, new_w), mode='bilinear', align_corners=False)
    x2u = F.interpolate(x2, size=(new_h, new_w), mode='bilinear', align_corners=False)
    flow = self.net(x1u, x2u, num_flow_updates=_RAFT_ITERS)[-1]
    flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
    flow[:, 0] *= w / new_w
    flow[:, 1] *= h / new_h
    return flow.to(img1.dtype)
```

**Additional fix:** Moved `self.net.float()` from `forward()` (called every pass) to
`__init__` (called once). Prevents AMP interference where `self.net` gets reset
to float32 on every forward pass.

**Original backup:** `raft_flow.py.bak2`

## 5. Verification

All tests pass (ran `verify_nan_fix.py` on server):

| Test | Result |
|------|--------|
| RAFT at 32x32 | OK (min=-1.35, max=2.88) |
| RAFT at 64x64 | OK (min=-11.39, max=6.09) |
| RAFT at 128x128 | OK (min=-24.19, max=12.29) |
| CFR forward with RAFT (512 LQ -> 64 downsample) | OK (min=-2.18, max=2.05) |

## 6. Smoke Test

⏳ Running (background task `bahfm8vgt`) — 15-step CFR-only RAFT training on GPUs 0,1
to verify end-to-end pipeline with the fix.

## 7. Related Files

| File | Role |
|------|------|
| `src/cross_frame_retrieval/raft_flow.py` | RAFT wrapper — FIXED |
| `src/cross_frame_retrieval/cfr_main.py` | CFR model, 8x downsample at line 645 |
| `verify_nan_fix.py` | Verification script for this fix |
| `diagnose_nan.py` | Original NaN diagnostic script |
| `runs/smoke_cfr_raft_v2/` | Smoke test output after fix |
