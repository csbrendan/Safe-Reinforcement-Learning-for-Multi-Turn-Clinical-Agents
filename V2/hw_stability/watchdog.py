"""Independent thermal/fault watchdog. Reads sensors itself (not the telemetry log, not the workload) and kills the
workload's whole process group when any limit trips.

    python V2/hw_stability/watchdog.py --pgid-file run/workload.pgid --log run/watchdog.log \
        --gpu-max 80 --acpi-max 82 --max-seconds 2400

Trips (any one): GPU temp >= --gpu-max; any ACPI zone >= --acpi-max; HW slowdown, SW thermal, HW thermal or HW
power-brake throttle bit active; a new kernel error line (Xid/NVRM/MCE/thermal/OOM/panic/hung task/hardware error);
GPU query failing 5 times in a row; wall clock over --max-seconds. SW power capping (0x4) is normal under load and is
only logged. After a trip it writes TRIPPED to --trip-file so the driver does not start the next stage.
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone

BAD_BITS = {0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}
KERR = re.compile(r"Xid|NVRM|mce|machine check|thermal|throttl|oom|out of memory|panic|hung task|hardware error|"
                  r"EDAC", re.I)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks_throttle_reasons.active",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
    t, w, m = [x.strip() for x in out.strip().split(",")]
    return float(t), float(w), int(m, 16)


def acpi_max() -> float:
    v = []
    for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            v.append(float(open(p).read()) / 1000)
        except Exception:
            pass
    return max(v) if v else 0.0


def kernel_errors(since: str) -> list[str]:
    out = subprocess.run(["journalctl", "-k", "--since", since, "--no-pager", "-q", "-o", "cat"],
                         capture_output=True, text=True, timeout=5).stdout
    return [l for l in out.splitlines() if KERR.search(l) and "Registered thermal governor" not in l]


def kill_group(pgid_file: str, log) -> None:
    try:
        pgid = int(open(pgid_file).read().strip())
    except Exception:
        log("no pgid file; nothing to kill")
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
            log(f"sent {sig.name} to pgid {pgid}")
        except ProcessLookupError:
            return
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgid-file", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--trip-file", default=None)
    ap.add_argument("--gpu-max", type=float, default=80.0)
    ap.add_argument("--gpu-warn", type=float, default=75.0)
    ap.add_argument("--acpi-max", type=float, default=82.0)
    ap.add_argument("--max-seconds", type=float, default=2400)
    a = ap.parse_args()
    trip_file = a.trip_file or a.log + ".TRIPPED"
    f = open(a.log, "a", buffering=1)

    def log(msg):
        f.write(f"{now()} {msg}\n")
        f.flush()
        os.fsync(f.fileno())

    since = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start, fails, warned = time.time(), 0, False
    log(f"watchdog start gpu_max={a.gpu_max} acpi_max={a.acpi_max} max_s={a.max_seconds}")
    while True:
        reason = None
        try:
            t, w, mask = gpu()
            fails = 0
        except Exception as e:  # noqa: BLE001
            fails += 1
            t, w, mask = -1, -1, 0
            if fails >= 5:
                reason = f"gpu query failed 5x ({e})"
        am = acpi_max()
        if t >= a.gpu_max:
            reason = f"gpu {t} C >= {a.gpu_max}"
        elif am >= a.acpi_max:
            reason = f"acpi {am} C >= {a.acpi_max}"
        elif bad := [n for b, n in BAD_BITS.items() if mask & b]:
            reason = f"throttle {','.join(bad)} mask=0x{mask:x}"
        elif time.time() - start > a.max_seconds:
            reason = f"wall clock > {a.max_seconds}s"
        if reason is None and int(time.time() - start) % 5 == 0:
            ke = kernel_errors(since)
            if ke:
                reason = f"kernel error: {ke[-1][:160]}"
        if not warned and t >= a.gpu_warn:
            warned = True
            log(f"WARN gpu {t} C >= {a.gpu_warn}")
        if reason:
            log(f"TRIP {reason} (gpu={t} C, {w} W, acpi_max={am} C)")
            open(trip_file, "w").write(f"{now()} {reason}\n")
            kill_group(a.pgid_file, log)
            log("watchdog exit after trip")
            return
        if not os.path.exists(a.pgid_file + ".running"):
            log("workload finished; watchdog exit")
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
