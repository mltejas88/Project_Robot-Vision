#!/usr/bin/env python3
"""
capture_on_key.py

Continuously reads camera frames. Save a captured frame when you press:
 - GUI mode: 'n' to capture, 'q' to quit
 - Headless mode: press Enter (or type 'n' + Enter) to capture, Ctrl-C to quit

Images are saved into --outdir with sequential names capture_0000.jpg, ...
"""
import cv2
import argparse
import os
import sys
import select
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cam', type=int, default=0, help='camera index (default 0)')
    p.add_argument('--outdir', type=str, default='captures', help='directory to save captures')
    p.add_argument('--prefix', type=str, default='capture', help='filename prefix')
    p.add_argument('--headless', action='store_true', help='run without GUI; press Enter in terminal to capture')
    p.add_argument('--fmt', type=str, default='jpg', choices=['jpg','png'], help='image format')
    return p.parse_args()

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def next_filename(outdir, prefix, idx, fmt):
    return os.path.join(outdir, f"{prefix}_{idx:04d}.{fmt}")

def main():
    args = parse_args()
    ensure_dir(args.outdir)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Failed to open camera {args.cam}", file=sys.stderr)
        return

    idx = 0
    # find starting index (avoid overwrite)
    while os.path.exists(next_filename(args.outdir, args.prefix, idx, args.fmt)):
        idx += 1

    print("Camera opened. Controls:")
    if args.headless:
        print(" Headless mode: press Enter (or type 'n' + Enter) to capture a frame. Ctrl-C to quit.")
    else:
        print(" GUI mode: window shown. Press 'n' to capture, 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera", file=sys.stderr)
                break

            if not args.headless:
                cv2.imshow("capture_on_key - press 'n' to save, 'q' to quit", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('n'):
                    fname = next_filename(args.outdir, args.prefix, idx, args.fmt)
                    cv2.imwrite(fname, frame)
                    print(f"[{datetime.now().isoformat()}] Saved {fname}")
                    idx += 1
                elif key == ord('q'):
                    print("Quit (q pressed).")
                    break
            else:
                # headless: print prompt once and poll stdin non-blocking
                # Use select to detect Enter keypress without blocking frame loop.
                # This keeps camera warm and responsive while waiting for user capture.
                rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
                if rlist:
                    line = sys.stdin.readline().strip()
                    # treat empty line (Enter) or 'n' as capture command
                    if line == "" or line.lower() == 'n':
                        fname = next_filename(args.outdir, args.prefix, idx, args.fmt)
                        cv2.imwrite(fname, frame)
                        print(f"[{datetime.now().isoformat()}] Saved {fname}")
                        idx += 1
                    else:
                        print(f"Ignored input: '{line}' (press Enter or 'n' to capture).")
                # small sleep is implicit in cap.read loop; loop will run fast
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl-C). Exiting.")
    finally:
        cap.release()
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        print(f"Done. {idx} images saved in '{args.outdir}'.")

if __name__ == "__main__":
    main()