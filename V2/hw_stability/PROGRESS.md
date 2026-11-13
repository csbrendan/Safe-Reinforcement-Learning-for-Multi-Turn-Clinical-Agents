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
