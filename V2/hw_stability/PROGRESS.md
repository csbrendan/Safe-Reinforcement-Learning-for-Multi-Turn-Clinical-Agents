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
