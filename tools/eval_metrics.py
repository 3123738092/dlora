"""Evaluate per-clip PSNR / SSIM / LPIPS on a VSR result directory.

Layout assumption:
  pred_dir/<clip>/frame_NNNN.png      (1-indexed, DLoRAL drops frame 0)
  gt_dir/<clip>/00000NNN.png          (0-indexed, official UDM10)

Frame mapping: pred frame_K.png is compared to gt 00000K.png.
"""
import argparse, csv, glob, os, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import lpips

def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)

def to_lpips_tensor(arr_uint8, device):
    t = torch.from_numpy(arr_uint8).float().permute(2,0,1) / 127.5 - 1.0
    return t.unsqueeze(0).to(device)

def gt_name_for(pred_frame_name: str) -> str:
    # frame_0001.png -> 00000001.png
    stem = Path(pred_frame_name).stem  # "frame_0001"
    idx = int(stem.split("_")[-1])
    return f"{idx:08d}.png"

def eval_clip(pred_clip_dir, gt_clip_dir, lpips_net, device):
    pred_files = sorted(glob.glob(os.path.join(pred_clip_dir, "frame_*.png")))
    if not pred_files:
        return None
    psnrs, ssims, lps = [], [], []
    for pf in pred_files:
        gf = os.path.join(gt_clip_dir, gt_name_for(pf))
        if not os.path.exists(gf):
            continue
        pred = load_rgb(pf); gt = load_rgb(gf)
        if pred.shape != gt.shape:
            # resize pred to gt size (some clips may have residual rounding)
            pred = np.asarray(Image.fromarray(pred).resize(
                (gt.shape[1], gt.shape[0]), Image.BICUBIC), dtype=np.uint8)
        psnrs.append(psnr_fn(gt, pred, data_range=255))
        ssims.append(ssim_fn(gt, pred, channel_axis=2, data_range=255))
        with torch.no_grad():
            d = lpips_net(to_lpips_tensor(pred, device), to_lpips_tensor(gt, device))
        lps.append(float(d.item()))
    if not psnrs:
        return None
    return dict(n=len(psnrs),
                psnr=float(np.mean(psnrs)),
                ssim=float(np.mean(ssims)),
                lpips=float(np.mean(lps)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir",   required=True)
    ap.add_argument("--out_csv",  default=None)
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    lp = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()

    pred_clips = sorted(d for d in os.listdir(args.pred_dir)
                        if os.path.isdir(os.path.join(args.pred_dir, d)))
    rows = []
    for c in pred_clips:
        pdir = os.path.join(args.pred_dir, c)
        gdir = os.path.join(args.gt_dir, c)
        if not os.path.isdir(gdir):
            print(f"[skip] no GT clip {c}", file=sys.stderr)
            continue
        r = eval_clip(pdir, gdir, lp, device)
        if r is None:
            print(f"[skip] no frames {c}", file=sys.stderr); continue
        n,p_,sm,lp_ = r["n"], r["psnr"], r["ssim"], r["lpips"]
        rows.append((c, n, p_, sm, lp_))
        print(f"{c:20s} n={n:3d} PSNR={p_:.3f} SSIM={sm:.4f} LPIPS={lp_:.4f}", flush=True)

    if not rows:
        print("no rows", file=sys.stderr); sys.exit(1)

    n_tot = sum(r[1] for r in rows)
    p_avg = sum(r[2]*r[1] for r in rows)/n_tot
    s_avg = sum(r[3]*r[1] for r in rows)/n_tot
    l_avg = sum(r[4]*r[1] for r in rows)/n_tot
    print("-"*70)
    print(f"{'WEIGHTED MEAN':20s} n={n_tot:3d} PSNR={p_avg:.3f} SSIM={s_avg:.4f} LPIPS={l_avg:.4f}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip","n_frames","psnr","ssim","lpips"])
            for r in rows: w.writerow(r)
            w.writerow(["__weighted_mean__", n_tot, p_avg, s_avg, l_avg])
        print(f"[csv] -> {args.out_csv}")

if __name__ == "__main__":
    main()
