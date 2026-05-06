"""GT-free OCR metrics on a VSR result directory.

Layout: pred_dir/<clip>/*.png
Metrics (per-clip + weighted mean):
    n_boxes      total detected text boxes
    avg_conf     mean confidence over all boxes
    n_chars      total recognized characters
    box/frame    detection density
Higher = more / clearer text recovered. No GT needed.
"""
import argparse, csv, glob, os, sys, warnings
warnings.filterwarnings('ignore')
import pyclipper  # noqa: F401  preload to dodge zlib symbol clash with paddle/cv2
from paddleocr import PaddleOCR

def run_clip(ocr, clip_dir, max_frames=None):
    files = sorted(glob.glob(os.path.join(clip_dir, '*.png')))
    if max_frames:
        files = files[:max_frames]
    if not files:
        return None
    n_box = 0
    conf_sum = 0.0
    n_char = 0
    for f in files:
        try:
            res = ocr.ocr(f, cls=True)
        except Exception as e:
            print('[err] {}: {}'.format(f, e), file=sys.stderr)
            continue
        if not res or res[0] is None:
            continue
        for line in res[0]:
            if line is None or len(line) < 2:
                continue
            txt = line[1][0]
            conf = float(line[1][1])
            n_box += 1
            conf_sum += conf
            n_char += len(txt)
    avg = (conf_sum / n_box) if n_box else 0.0
    return {
        'n_frames': len(files),
        'n_box': n_box,
        'avg_conf': avg,
        'n_char': n_char,
        'box_per_frame': n_box / max(len(files), 1),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--out_csv', default=None)
    ap.add_argument('--lang', default='ch', help='ch|en|...')
    ap.add_argument('--use_gpu', action='store_true')
    ap.add_argument('--max_frames', type=int, default=None,
                    help='Cap frames per clip (debug)')
    args = ap.parse_args()

    ocr = PaddleOCR(use_angle_cls=True, lang=args.lang,
                    use_gpu=args.use_gpu, show_log=False)

    clips = sorted(d for d in os.listdir(args.pred_dir)
                   if os.path.isdir(os.path.join(args.pred_dir, d)))
    rows = []
    for c in clips:
        r = run_clip(ocr, os.path.join(args.pred_dir, c), args.max_frames)
        if r is None:
            continue
        nf = r['n_frames']; nb = r['n_box']; ac = r['avg_conf']
        nc = r['n_char']; bf = r['box_per_frame']
        rows.append((c, nf, nb, ac, nc, bf))
        line = '{:20s} frames={:3d}  boxes={:5d}  avg_conf={:.3f}  chars={:5d}  box/f={:.2f}'.format(
            c, nf, nb, ac, nc, bf)
        print(line, flush=True)

    if not rows:
        print('no rows', file=sys.stderr)
        sys.exit(1)

    nf = sum(r[1] for r in rows)
    nb = sum(r[2] for r in rows)
    nc = sum(r[4] for r in rows)
    avg_conf = sum(r[3] * r[2] for r in rows) / max(nb, 1)
    print('-' * 78)
    print('{:20s} frames={:3d}  boxes={:5d}  avg_conf={:.3f}  chars={:5d}  box/f={:.2f}'.format(
        'MEAN', nf, nb, avg_conf, nc, nb / max(nf, 1)))

    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['clip', 'n_frames', 'n_box', 'avg_conf', 'n_char', 'box_per_frame'])
            for r in rows:
                w.writerow(r)
            w.writerow(['__mean__', nf, nb, avg_conf, nc, nb / max(nf, 1)])
        print('[csv] -> {}'.format(args.out_csv))

if __name__ == '__main__':
    main()
