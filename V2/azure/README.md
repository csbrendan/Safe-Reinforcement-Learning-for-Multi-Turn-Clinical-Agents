# V2 Azure GPU VM: setup checklist

Scripts for the day GPU quota lands. Written 2026-11-22; **not yet run**, because GPU quota is still 0.

| File | Runs on | Does |
|---|---|---|
| `vm.sh` | Spark 2 | `create`, `push`, `setup`, `ssh`, `status`, `ip`, `delete`. `create` and `delete` refuse unless `APPROVED=yes` |
| `setup_remote.sh` | the VM | driver check, SkyRL env at the pinned commit, MedGemma 27B download, CPU tests, GPU check; trains nothing |

## Before quota lands (free)

- [ ] GPU quota granted for NCads H100 v5 in **West US 3 or West US 2** (H100 is `NotAvailableForSubscription` in
      eastus2, eastus and southcentralus for this subscription).
- [x] Hugging Face license for `google/medgemma-27b-text-it` accepted (2026-11-22); Spark 2's token has access.
- [ ] SD-Bench clone present at `~/v2_bench/sdbench`, including the chargemaster xlsx.
- [ ] For MIMIC on the VM: PhysioNet DUA cloud-storage terms checked and the Azure OpenAI human-review opt-out
      approved. Until then the VM runs synthetic cases only.

## On the day (needs Brendan's go-ahead: billing starts at `create`)

```bash
cd ~/rl-gains-in-clinical-tasks
APPROVED=yes bash V2/azure/vm.sh create      # ~5–10 min; 1x H100 NVL, 1 TB Premium SSD, SSH from Spark 2 only
bash V2/azure/vm.sh push                     # SD-Bench clone, repo, azure.env, HF token (copied, never printed)
bash V2/azure/vm.sh setup                    # ~20–40 min (SkyRL env + 54 GB of weights)
```

Then the pilot, `V2/skyrl/run_pilot_27b.sh` (see [`../skyrl_plan.md`](../skyrl_plan.md)), once we've agreed on
it together.

## Holding the GPU

- On-demand (`PRIORITY=Regular`, the default) for training. Spot can be taken back with 30 seconds' notice, so use it
  only for restartable evaluation work.
- During the run window, keep the VM allocated. "Stop (deallocate)" releases the GPU, and restarting can fail if the
  region is out of capacity.
- At the end: `APPROVED=yes bash V2/azure/vm.sh delete` removes the VM, disk and IP.
- Cost: 1x H100 NVL on-demand ≈ $6.98/h ≈ $168/day, plus about $5/day for the disk. The VM bills while allocated, idle or not.

## Assumptions to check on first boot

- The Azure HPC image (`microsoft-dsvm:ubuntu-hpc:2404`) ships an NVIDIA driver ≥ 580, which SkyRL's torch 2.14 /
  CUDA 13 wheels need. `setup_remote.sh` stops if it doesn't.
- `--security-type Standard` (no Secure Boot), so the NVIDIA kernel modules load.
- MedGemma 27B fits next to vLLM on one H100 NVL with sleep mode (estimate 70–80 GB). The pilot measures it.
