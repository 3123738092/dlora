#!/usr/bin/env python3
"""Diagnose NaN source in CFR-only training with RAFT flow.

Runs a single CFR forward pass with both SPyNet and RAFT, comparing
intermediate tensor statistics to find where NaN first appears.
"""
import os, sys
os.environ['TRANSFORMERS_OFFLINE'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

sys.path.insert(0, '/nas-files/yuhanchen/lora/DLoRAL/src')

from src.cross_frame_retrieval.cfr_main import CFR_model, SPyNet
from src.cross_frame_retrieval.raft_flow import RAFTFlowEstimator


def stats(name, t):
    """Return a one-line min/max/mean/NaN/Inf summary for tensor t."""
    if not isinstance(t, torch.Tensor):
        return f"{name}: not a tensor ({type(t).__name__})"
    if t.numel() == 0:
        return f"{name}: empty"
    x = t.detach().float()
    nan_flag = torch.isnan(x).any().item()
    inf_flag = torch.isinf(x).any().item()
    return (f"{name}: shape={list(t.shape)} dtype={t.dtype} "
            f"min={x.min().item():.6f} max={x.max().item():.6f} "
            f"mean={x.mean().item():.6f} "
            f"{'!!!NaN!!!' if nan_flag else ''}{'!!!Inf!!!' if inf_flag else ''}")


def make_diagnostic_forward(cfr_net):
    """Monkey-patch CFR.forward to print stats at every layer."""
    orig_forward = cfr_net.forward
    orig_compute_flow = cfr_net.compute_flow

    def debug_compute_flow(lqs):
        print("--- compute_flow ---")
        print(stats("  lqs", lqs))
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
        print(stats("  lqs_1 (ref)", lqs_1))
        print(stats("  lqs_2 (supp)", lqs_2))
        flows_backward = cfr_net.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_forward = cfr_net.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        print(stats("  flows_forward", flows_forward))
        print(stats("  flows_backward", flows_backward))
        return flows_forward, flows_backward

    def debug_forward(lqs, vae_feat, uncertainty_map, external_flows=None):
        print("=" * 70)
        print("CFR diagnostic forward pass")
        print("=" * 70)
        n, t, c, h, w = lqs.size()
        n_feat, t_feat, c_feat, h_feat, w_feat = vae_feat.size()

        print(stats("lqs", lqs))
        print(stats("vae_feat", vae_feat))
        print(stats("uncertainty_map", uncertainty_map))

        lqs_downsample = F.interpolate(
            lqs.view(-1, c, h, w), scale_factor=0.125,
            mode='bicubic').view(n, t, c, h // 8, w // 8)
        print(stats("lqs_downsample", lqs_downsample))

        feats_spatial = vae_feat.clone()

        if external_flows is not None:
            flows_forward, flows_backward = external_flows
            print("--- Using external flows ---")
            print(stats("  flows_forward", flows_forward))
            print(stats("  flows_backward", flows_backward))
        else:
            flows_forward, flows_backward = debug_compute_flow(lqs_downsample)

        fused_features = feats_spatial.clone()
        aligned_feat_return = feats_spatial.clone()
        weight_map = torch.ones((n_feat, t_feat, 1, h_feat, w_feat),
                                dtype=feats_spatial.dtype,
                                device=feats_spatial.device)

        for t_index in range(1, t_feat):
            print(f"\n--- Frame {t_index} ---")
            cur_feat = feats_spatial[:, t_index]
            prev_feat = feats_spatial[:, t_index - 1]
            flow_cur = flows_forward[:, t_index - 1, :, :, :]

            print(stats(f"  cur_feat [{t_index}]", cur_feat))
            print(stats(f"  prev_feat [{t_index-1}]", prev_feat))
            print(stats(f"  flow [{t_index-1}]", flow_cur))

            # Warp previous feature
            aligned_prev_feat = cfr_net._flow_warp_fn(
                prev_feat, flow_cur.permute(0, 2, 3, 1))
            print(stats("  aligned_prev_feat", aligned_prev_feat))

            # Warp previous image
            cur_img = lqs_downsample[:, t_index]
            prev_img = lqs_downsample[:, t_index - 1]
            print(stats(f"  cur_img [{t_index}]", cur_img))
            print(stats(f"  prev_img [{t_index-1}]", prev_img))
            aligned_prev_img = cfr_net._flow_warp_fn(
                prev_img, flow_cur.permute(0, 2, 3, 1))
            print(stats("  aligned_prev_img", aligned_prev_img))

            # Cross-attention
            print("--- cross_attn_module ---")
            fused_features[:, t_index], out_feat_down = cfr_net.cross_attn_module(
                cur_feat, aligned_prev_feat, cur_feat, aligned_prev_feat)
            print(stats(f"  fused_features[:,{t_index}]", fused_features[:, t_index]))
            print(stats("  out_feat_down", out_feat_down))

            weight_map[:, t_index] = uncertainty_map[:, t_index - 1]
            aligned_feat_return[:, t_index] = aligned_prev_feat

        print(stats("fused_features (final)", fused_features))
        has_nan = torch.isnan(fused_features).any().item()
        if has_nan:
            print("!!! NaN DETECTED in fused_features !!!")
            # Backtrack to find first NaN in loop
        else:
            print("--- No NaN in CFR forward output ---")
        return fused_features, weight_map, aligned_feat_return

    cfr_net.forward = debug_forward
    # Expose flow_warp for the debug wrapper
    from src.cross_frame_retrieval.cfr_main import flow_warp
    cfr_net._flow_warp_fn = flow_warp
    return cfr_net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_estimator', type=str, default='raft',
                        choices=['spynet', 'raft'])
    flag = parser.parse_args()

    print(f"=== NaN Diagnostic: flow_estimator = {flag.flow_estimator} ===")
    print("Running in float32 (no AMP) for clean diagnosis\n")

    cfr = CFR_model(
        mid_channels=64, num_blocks=7, is_low_res_input=False,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth',
        flow_estimator=flag.flow_estimator)
    cfr = cfr.cuda()
    cfr.eval()

    # Try loading CFR checkpoint
    ckpt_path = '/nas-files/yuhanchen/lora/DLoRAL/preset/models/checkpoints/model.pkl'
    if os.path.exists(ckpt_path):
        print(f"\nLoading CFR checkpoint from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location='cuda')
        cfr_sd = {}
        for k, v in sd.items():
            if 'cfr_main_net' in k:
                new_k = k.replace('cfr_main_net.', '')
                cfr_sd[new_k] = v
        if cfr_sd:
            missing, unexpected = cfr.load_state_dict(cfr_sd, strict=False)
            print(f"  Loaded {len(cfr_sd)} CFR params "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print("  No CFR params in checkpoint, using random init")
    else:
        print(f"Checkpoint not found, using random init")

    flow_params = sum(p.numel() for p in cfr.flow_net.parameters())
    print(f"Flow estimator: {flag.flow_estimator} ({flow_params:,} params)")

    # Patch forward for diagnostics
    from src.cross_frame_retrieval.cfr_main import flow_warp
    cfr._flow_warp_fn = flow_warp
    orig_forward = cfr.forward
    orig_compute_flow = cfr.compute_flow

    # ---- SPyNet baseline ----
    print("\n" + "=" * 70)
    print("PART 1: SPyNet baseline (for comparison)")
    print("=" * 70)
    B, T = 1, 2  # just 2 frames for quick test
    H, W = 512, 512
    Hl, Wl = 64, 64
    torch.manual_seed(42)
    lqs_test = torch.rand(B, T, 3, H, W).cuda()
    feat_test = torch.randn(B, T, 4, Hl, Wl).cuda() * 0.5
    uncert_test = torch.rand(B, T, 1, Hl, Wl).cuda()

    # Use 64x64 LQ inputs (matches CFR internal lqs_downsample)
    lqs_ds_test = F.interpolate(lqs_test.view(-1, 3, H, W),
                                 scale_factor=0.125, mode='bicubic')
    lqs_ds_test = lqs_ds_test.view(B, T, 3, Hl, Wl)
    lqs_flat_64 = lqs_ds_test[:, :2].reshape(-1, 3, Hl, Wl)
    img1_64 = lqs_flat_64[0:1]
    img2_64 = lqs_flat_64[1:2]

    spynet_ref = SPyNet(
        pretrained='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth')
    spynet_ref = spynet_ref.cuda().eval()
    with torch.no_grad():
        flow_spynet = spynet_ref(img1_64, img2_64)
    print(stats("SPyNet flow (64x64)", flow_spynet))

    # Run CFR forward with SPyNet flow
    flow_fwd_spynet = flow_spynet.unsqueeze(1)
    flow_bwd_spynet = spynet_ref(img2_64, img1_64).unsqueeze(1)

    with torch.no_grad():
        out_spynet = orig_forward(lqs_test, feat_test, uncert_test,
                                   external_flows=(flow_fwd_spynet, flow_bwd_spynet))
    print(stats("CFR output (SPyNet flow)", out_spynet[0]))

    # ---- RAFT diagnostic ----
    print("\n" + "=" * 70)
    print(f"PART 2: Full CFR forward with {flag.flow_estimator.upper()} flow")
    print("=" * 70)
    raft = RAFTFlowEstimator(pretrained=True, small=True).cuda().eval()
    with torch.no_grad():
        flow_raft = raft(img1_64, img2_64)
    print(stats("RAFT flow (64x64)", flow_raft))

    # Compare flows
    diff = (flow_raft - flow_spynet).abs()
    cos = F.cosine_similarity(flow_raft.flatten().unsqueeze(0),
                               flow_spynet.flatten().unsqueeze(0))
    print(f"Flow diff: min={diff.min():.6f} max={diff.max():.6f} mean={diff.mean():.6f}")
    print(f"Flow cos-sim: {cos.item():.6f}")
    print(f"RAFT  range: [{flow_raft.min():.4f}, {flow_raft.max():.4f}]")
    print(f"SPyNet range: [{flow_spynet.min():.4f}, {flow_spynet.max():.4f}]")

    # Now run full debug forward
    flow_fwd_raft = flow_raft.unsqueeze(1)
    flow_bwd_raft = raft(img2_64, img1_64).unsqueeze(1)

    # Re-init CFR with clean state for RAFT test
    cfr2 = CFR_model(
        mid_channels=64, num_blocks=7, is_low_res_input=False,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth',
        flow_estimator=flag.flow_estimator)
    cfr2 = cfr2.cuda().eval()
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location='cuda')
        cfr_sd = {k.replace('cfr_main_net.', ''): v
                  for k, v in sd.items() if 'cfr_main_net' in k}
        if cfr_sd:
            cfr2.load_state_dict(cfr_sd, strict=False)
    cfr2._flow_warp_fn = flow_warp

    # Detailed per-layer check
    print("\n--- Per-layer NaN check ---")
    lqs_ds = F.interpolate(lqs_test.view(-1, 3, H, W),
                            scale_factor=0.125, mode='bicubic')
    lqs_ds = lqs_ds.view(B, T, 3, Hl, Wl)

    # feat_extract
    print("\n[1] feat_extract")
    feat_out = cfr2.feat_extract(feat_test.view(-1, 4, Hl, Wl))
    print(stats("  feat_extract output", feat_out))

    # flow_warp with RAFT flow
    print("\n[2] flow_warp (with RAFT flow)")
    prev_feat = feat_test[:, 0]
    aligned = flow_warp(prev_feat, flow_raft.permute(0, 2, 3, 1))
    print(stats("  warped feat", aligned))
    prev_img = lqs_ds[:, 0]
    aligned_img = flow_warp(prev_img, flow_raft.permute(0, 2, 3, 1))
    print(stats("  warped img", aligned_img))

    # cross_attn_module
    print("\n[3] cross_attn_module")
    cur_feat = feat_test[:, 1]
    cur_img = lqs_ds[:, 1]
    fused, out_down = cfr2.cross_attn_module(cur_feat, aligned_feat=aligned,
                                              second_frame_feat=cur_feat,
                                              first_frame_feat_aligned=aligned)
    print(stats("  cross_attn fused", fused))
    print(stats("  cross_attn out_down", out_down))

    # weight_predict
    print("\n[4] weight_predict")
    # weight_predict takes 3-channel input (uncertainty-like)
    wp_in = torch.randn(B, 3, Hl, Wl).cuda() * 0.5
    wp_out = cfr2.weight_predict(wp_in)
    print(stats("  weight_predict output", wp_out))

    # reconstruction_feat
    print("\n[5] reconstruction_feat")
    recon_out = cfr2.reconstruction_feat(fused.view(-1, fused.size(1), Hl, Wl))
    print(stats("  reconstruction_feat output", recon_out))

    # conv_hr + conv_last
    print("\n[6] conv_hr + conv_last")
    hr = cfr2.conv_hr(recon_out)
    hr = cfr2.lrelu(hr)
    print(stats("  conv_hr output", hr))
    last = cfr2.conv_last(hr)
    print(stats("  conv_last output", last))

    # Summary
    has_nan = False
    for name, t in [("feat_extract", feat_out), ("warped_feat", aligned),
                     ("warped_img", aligned_img), ("cross_attn_fused", fused),
                     ("reconstruction_feat", recon_out), ("conv_hr", hr),
                     ("conv_last", last)]:
        if torch.isnan(t).any():
            print(f"\n*** FIRST NaN DETECTED at: {name} ***")
            has_nan = True
            break
    if not has_nan:
        print("\n*** No NaN detected in any layer (fp32 mode) ***")
        print("NaN in training may be caused by fp16 overflow — "
              "RAFT flow likely has larger magnitude than SPyNet, "
              "causing Q*K^T similarity overflow in cross_attn_module under fp16.")


if __name__ == '__main__':
    main()
