# V2 hardware stability — running log (2026-11-13 overnight)

Written incrementally so an abrupt Spark 1 power-off cannot lose it. Spark 1 telemetry is also mirrored live to
Spark 2 at `~/v2_hw/<run>/spark1_mirror.csv`.

## Pre-test findings (03:43–03:55 UTC)
- Both Sparks rebooted at 03:43 UTC (user's electrical change). Both reachable; nothing on either GPU.
- Spark 1 journal: all 7 previous boots end abruptly (no shutdown sequence, no kernel panic/thermal/Xid/OOM/MCE
  lines; pstore empty). Spark 2 boots also all end without a shutdown sequence, so an abrupt end alone does not
  distinguish a fault from a pulled plug. Spark 2 boot -2 ended 2026-11-08 17:10 UTC, ~5 min after Spark 1's boot -5
  ended (17:05) — possibly a shared event or a manual power cycle; not determinable from logs.
- Idle (P8, ~5 W GPU, load < 0.3): Spark 1 GPU 55–57 C, ACPI zones 58–60 C, NVMe 55 C, NIC 64 C.
  Spark 2 GPU 43–44 C, ACPI 44–46 C, NVMe 43 C, NIC 49–50 C. Spark 1 idles ~13–15 C hotter at the same load.
- GPU slowdown limit (current temp + T.Limit headroom): Spark 1 ≈ 95 C, Spark 2 ≈ 97 C. The earlier crashes at 88 C
  happened below the GPU's own slowdown point with no throttle event logged.

## Test 1: Spark 1 progressive ramp (03:53–03:56 UTC) — STOPPED at the thermal limit
`results/v2_hw/ramp1/` (Spark 1 local log, plus the mirror copy written on Spark 2).
- The first stage ("25% duty" bf16 matmul) actually ran the GPU at its power cap (~49–51 W, SW-power-cap bit set)
  on both machines, so the stage labels don't map to power. Every stage was effectively full GPU power.
- Spark 1, from load start: ACPI max 60.9 → 74.6 (15 s) → 78.1 (30 s) → 83.3 (45 s) → **85.3 °C (55 s)**, still
  rising. GPU reading peaked at 81 °C at about 35 s. NIC 64 → 76 °C. The watchdog killed the workload at 03:54:33
  (ACPI 82.8 °C on its own read). No crash, no throttle bit, no kernel error.
- **Harness bug found:** the GPU reading was 80–81 °C for about 8 s (03:54:08–16) and the GPU limit (80) did not
  trip. The watchdog ran nvidia-smi as a subprocess, and nvidia-smi stalled under load (telemetry also has an 11 s
  gap, 03:53:49–54:00). Fixed: it now reads NVML in-process, logs every 0.5 s sample, and trips when the sensor stops
  answering while hot. The ACPI limit was what stopped this run.
- Per the stop rule (approaching 85 °C), no further stress on Spark 1 and **no 20–30 min realistic test on
  Spark 1**.

## Control: identical stage on Spark 2 (03:58–04:00 UTC), 120 s, completed
`results/v2_hw/spark2_ctl/`. Same ~47 W. ACPI max 44.8 → 58.9 (15 s) → 63.1 (30 s) → 66.3 (45 s) → 68.8 (55 s) →
70.3 °C (115 s), flattening out. GPU peaked at 72 °C. Spark 2 idled at 43 °C GPU throughout Spark 1's test.

| t from load start | Spark 1 ACPI max | Spark 2 ACPI max |
|---|---|---|
| idle | 60.9 | 44.8 |
| +15 s | 74.6 | 58.9 |
| +30 s | 78.1 | 63.1 |
| +55 s | 85.3 (stopped) | 68.8 |
| +115 s | – | 70.3 |

**Reading.** Both machines heat up at the same rate in the first minute (+24 °C), but Spark 1 starts about 16 °C
hotter. The offset shows up on every sensor, including the NVMe (55 vs 42 °C idle) and the NIC (64 vs 50), so the
whole chassis is warmer. That points to Spark 1's surroundings or airflow (inlet air temperature, position or
recirculated exhaust, or a fan running slower) rather than to its power supply. It is consistent with the four
earlier power-offs at about 88 °C under load. The cause is not confirmed: the fan speed isn't exposed to the OS.
The GPU temperature reading drops by 10–13 °C mid-load on both machines while power stays flat, so the ACPI zones
are the more trustworthy limit.

**Electrical change.** Spark 1 did not power off, but the test stopped at 85 °C, below the ~88 °C where it shut
down before. So this neither shows that the electrical change helped nor rules out power events. No crash in a
1-minute test does not show there are no brownouts.

## Spark 2 under batched inference (04:05–05:10 UTC)
Qwen3-8B (:8101, 0.45) + MedGemma (:8100, 0.20) on Spark 2 under `watchdog.py` (GPU or ACPI ≥ 83 °C).
- **Trip 1, 04:19:54:** 8 requests × n = 32 = 256 MedGemma sequences in flight. GPU ~73 W, ACPI 55.6 → 83.0 °C in
  ~30 s. Servers killed by the watchdog. No kernel errors. Logs: Spark 2 `~/v2_hw/serve_trip1/`.
- Cut the load to 32 sequences: ~40 W GPU, ACPI steady at 75–80 °C.
- **Trip 2, 04:37:54:** ACPI 84.7 °C (peak 85.7 in telemetry) with the GPU at only 13.7 W, between two probe
  batches. Cause: a CPU burst. The first constrained-decoding request made vLLM compile my regex, which had large
  counted repeats ({0,1500}). That costs 3.0 s on 8 threads, against 0.1 s for the equivalent unbounded regex.
  The SoC zone rose ~10 °C in ~2 s. Fixed by using an unbounded regex. Logs: `~/v2_hw/serve_trip2/`.
- So Spark 2 also reaches the low 80s ACPI under heavy load (high GPU power or an all-core CPU burst). Its idle
  baseline is ~16 °C lower than Spark 1's. Neither trip was a crash or a hardware error; both were my 83 °C limit.
- Decode throughput (MedGemma, 32 sequences): ~150 tokens/s total (~5 tokens/s per sequence), the same rate as the
  11-11 runs (~90 tokens/s at 16 sequences).
- **Trip 3, 05:24:52 (my error):** I let the end of the first-turn probe overlap the start of the full-episode
  runs. GPU power went 44 → 61 → 83 W within 10 s, and ACPI went 77.9 → 83.0 °C. The watchdog fired at 83.0, but
  ACPI read **88.8 °C** 3 s later while the servers were still exiting on SIGTERM (5 s grace). Nothing crashed and
  nothing logged an error, but that is past the 85 °C rule. Changes:
  - kill grace 1 s, then SIGKILL;
  - server limits lowered to 80 °C GPU and 80 °C ACPI;
  - only one GPU workload at a time on Spark 2.
  Logs: Spark 2 `~/v2_hw/serve_trip3/`.
- **What Spark 2 tolerates:** about 45 W of GPU power settles at 75–80 °C ACPI. 60–85 W reaches 83–89 °C within
  10–30 s. Spark 1 already reached 85 °C at 50 W. Neither machine can run at full GPU power for long within an
  80–85 °C limit, so throughput plans must budget around 40–45 W.
