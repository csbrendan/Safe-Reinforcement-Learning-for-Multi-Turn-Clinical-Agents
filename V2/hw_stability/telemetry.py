"""1 Hz hardware telemetry for a DGX Spark, as CSV on stdout (flushed every line).

Pipe it to a local file and/or over SSH to the other Spark so the record survives an abrupt power loss:
    python V2/hw_stability/telemetry.py | tee -a local.csv | ssh 192.168.100.2 'cat >> remote.csv'

Columns: UTC time, GPU temp/power/SM clock/util, active clock-event (throttle) bitmask, the cumulative
SW-power-cap / HW-thermal / HW-power-brake counters (us), max and mean ACPI zone temp, NVMe composite, max mlx5 NIC
temp, 1-min load, used memory (GiB), and a count of new kernel error lines since start (Xid, MCE, thermal, OOM,
panic, hung task, hardware error).
"""
from __future__ import annotations
import glob
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

Q = "temperature.gpu,power.draw,clocks.sm,utilization.gpu,clocks_throttle_reasons.active"
KERR = re.compile(r"Xid|NVRM|mce|machine check|thermal|throttl|oom|out of memory|panic|hung task|hardware error|"
                  r"EDAC|segfault|watchdog", re.I)
HEADER = ("utc,gpu_c,gpu_w,sm_mhz,util,throttle_mask,swpower_us,hwthermal_us,hwbrake_us,acpi_max_c,acpi_mean_c,"
          "nvme_c,nic_max_c,load1,mem_used_gib,kerr_new")


def read(p: str) -> float | None:
    try:
        return float(open(p).read().strip())
    except Exception:
        return None


def counters() -> tuple[str, str, str]:
    try:
        out = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return "", "", ""
    def g(name):
        m = re.search(name + r"\s*:\s*(\d+)\s*us", out)
        return m.group(1) if m else ""
    return g("SW Power Capping"), g("HW Thermal Slowdown"), g("HW Power Braking")


def gpu() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={Q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return [x.strip() for x in out.split(",")]
    except Exception:
        return ["", "", "", "", ""]


def kernel_lines(since: str) -> int:
    try:
        out = subprocess.run(["journalctl", "-k", "--since", since, "--no-pager", "-q", "-o", "cat"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return -1
    return sum(1 for line in out.splitlines() if KERR.search(line) and "Registered thermal governor" not in line)


def mem_used_gib() -> float:
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0])
    return (m["MemTotal"] - m["MemAvailable"]) / 2**20


def main():
    since = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(HEADER, flush=True)
    i = 0
    cnt = ("", "", "")
    kerr = 0
    while True:
        t0 = time.time()
        g = gpu()
        if i % 5 == 0:
            cnt = counters()
            kerr = kernel_lines(since)
        acpi = [v / 1000 for v in (read(p) for p in glob.glob("/sys/class/thermal/thermal_zone*/temp")) if v]
        nvme = next((read(p) for p in glob.glob("/sys/class/hwmon/hwmon*/temp1_input")
                     if open(p.rsplit("/", 1)[0] + "/name").read().strip() == "nvme"), None)
        nic = [read(p) for p in glob.glob("/sys/class/hwmon/hwmon*/temp1_input")
               if open(p.rsplit("/", 1)[0] + "/name").read().strip() == "mlx5"]
        nic = [v for v in nic if v]
        row = [datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), *g, *cnt,
               f"{max(acpi):.1f}" if acpi else "", f"{sum(acpi) / len(acpi):.1f}" if acpi else "",
               f"{nvme / 1000:.1f}" if nvme else "", f"{max(nic) / 1000:.1f}" if nic else "",
               f"{open('/proc/loadavg').read().split()[0]}", f"{mem_used_gib():.1f}", str(kerr)]
        try:
            print(",".join(row), flush=True)
        except BrokenPipeError:
            sys.exit(0)
        i += 1
        time.sleep(max(0.0, 1.0 - (time.time() - t0)))


if __name__ == "__main__":
    main()
