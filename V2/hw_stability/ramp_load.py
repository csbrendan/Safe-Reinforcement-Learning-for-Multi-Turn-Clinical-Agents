"""Progressive GPU load for the stability test: bf16 matmul bursts at a set duty cycle per stage.

    python V2/hw_stability/ramp_load.py --stages 0.25:120,0.5:120,0.75:120,1.0:180 --self-max 78

Each 1 s window runs matmuls for `duty` seconds and sleeps the rest. The workload also checks the GPU temperature
every 2 s and exits on its own at --self-max (a third layer under the watchdog and the shell timeout). It prints one
line per stage with achieved TFLOP/s.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time

import torch


def gpu_temp() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=5).stdout
    return float(out.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="0.25:120,0.5:120,0.75:120,1.0:180")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--self-max", type=float, default=78.0)
    a = ap.parse_args()
    stages = [(float(d), float(s)) for d, s in (x.split(":") for x in a.stages.split(","))]
    x = torch.randn(a.n, a.n, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(a.n, a.n, device="cuda", dtype=torch.bfloat16)
    flop = 2 * a.n ** 3
    last_check = 0.0
    for duty, secs in stages:
        t_end = time.time() + secs
        n_mm, busy = 0, 0.0
        print(f"[ramp] {time.strftime('%H:%M:%S', time.gmtime())} stage duty={duty} for {secs:.0f}s", flush=True)
        while time.time() < t_end:
            w0 = time.time()
            while time.time() - w0 < duty:
                for _ in range(4):
                    torch.matmul(x, y)
                torch.cuda.synchronize()
                n_mm += 4
            busy += time.time() - w0
            rest = 1.0 - (time.time() - w0)
            if rest > 0:
                time.sleep(rest)
            if time.time() - last_check > 2:
                last_check = time.time()
                t = gpu_temp()
                if t >= a.self_max:
                    print(f"[ramp] SELF-STOP gpu {t} C >= {a.self_max}", flush=True)
                    sys.exit(3)
        print(f"[ramp] stage duty={duty} done: {n_mm * flop / max(busy, 1e-9) / 1e12:.1f} TFLOP/s while busy",
              flush=True)


if __name__ == "__main__":
    main()
