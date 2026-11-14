# Spark 1 backup and restore

How Spark 1's data is backed up, and how to rebuild a working Spark 1 from the backup if the unit is lost,
replaced under RMA or wiped. This file lives in git at `V2/hw_stability/BACKUP_RESTORE.md` and is also copied as
`README.md` into every backup folder.

**Spark 1:** hostname `spark-b4e7`, direct-link IP `192.168.100.1`, user `brendan`. It holds the repo, the
training environment and all checkpoints.
**Spark 2:** `spark-04af`, `192.168.100.2`. It serves the models and holds the backup.

## 1. What is backed up, and where

First backup taken **2026-11-14**, before a firmware update and a possible RMA.

| Backup location | Contents |
|---|---|
| Spark 2: `~/spark1_backup/` | **Everything** in the table below. Checksum-verified against Spark 1 on 2026-11-14. |
| Mac mini (`BRENDANs-Mac-mini.local`, user `brendanmurphy`): `~/Desktop/spark1_backup/` | All trained MedGemma checkpoints: 28 folders, 929 files, 56.1 GB, checksum-verified 2026-11-14. Also `results/`, the warranty files, `.secrets/`, `hf_token`, `claude_memory/`, `v2_bench/` without its venvs, the package list and this README. **Left out:** `checkpoints/*_merged` (rebuild with `framework/rl/merge_adapter.py` from the matching SFT adapter plus the base) and `checkpoints/medgemma4b_text` (rebuild with `framework.infra.prep_text_base` from `google/medgemma-4b-it`), plus `spark1/` (Linux-only venv). |

**Hard links:** `v2_bench` uses hard-linked files, and rsync without `-H` stores them as separate copies. So
`du` shows the backup ~0.25 GB larger than the original, even though the contents are identical.

| Path in the backup | Restores to (on Spark 1) | Size | What it is |
|---|---|---|---|
| `rl-gains-in-clinical-tasks/checkpoints/` | `~/rl-gains-in-clinical-tasks/checkpoints/` | ~89 GB | **Irreplaceable.** All LoRA adapters and intra-run checkpoints for V1 and V2 (`dh_*`, `dx_*`, `ti_*`, `grpo_rx_*`, `sft_rx_*`, `procreward*`, `dream_*`), plus the text-only base model `medgemma4b_text` |
| `rl-gains-in-clinical-tasks/results/` | `~/rl-gains-in-clinical-tasks/results/` | ~49 MB | Run outputs, evaluation logs, hardware telemetry (gitignored, so not on GitHub) |
| `rl-gains-in-clinical-tasks/V2/hw_stability/spark1_*.md` | same path | small | Warranty report and support ticket (deliberately not in git: they contain serial numbers) |
| `spark1/` | `~/spark1/` | ~11 GB | The Python environment `~/spark1/.venv` (torch, vllm, trl, peft, transformers) plus its README and earlier `pip freeze` files |
| `v2_bench/` | `~/v2_bench/` | ~1.1 GB | V2 benchmark checkouts and data (SD-Bench, MedMarks, simple-evals with its venv, HealthBench data), MedGemma generation-config variants, server logs |
| `.secrets/` | `~/.secrets/` | tiny | `anthropic.env` (API key). **Contains a credential: keep the backup private.** |
| `hf_token` | `~/.cache/huggingface/token` | tiny | Hugging Face login token, needed to download the gated MedGemma model |
| `claude_memory/` | `~/.claude/projects/-home-brendan-rl-gains-in-clinical-tasks/memory/` | small | Claude Code's project memory (notes on gotchas, decisions, hardware findings) |
| `venv_freeze_2026-11-14.txt` | – | small | Exact package versions of `~/spark1/.venv`, for rebuilding it from scratch |

**Not in the backup, and how to get each one back:**
- **The code** is all on GitHub: `git@github.com:csbrendan/Safe-Reinforcement-Learning-for-Multi-Turn-Clinical-Agents.git`.
  Anything uncommitted is lost, so push before relying on this backup.
- **The Hugging Face model cache** (`~/.cache/huggingface`, ~26 GB) can be downloaded again. Spark 2 already keeps
  serving copies in `~/v2_bench/models/` (`medgemma4b_text`, `medgemma4b_text_v2stop`, `qwen3-8b`, `qwen3-8b-base`).
- **SSH private keys** are not copied; create new ones (step 3).
- **System settings and the OS** come from a fresh DGX OS install plus steps 2–3.

## 2. Restoring onto a replacement or wiped Spark

Run these on the new Spark 1 unless a step says otherwise. They assume the same username, `brendan`. Keep it,
because the Python venv has `/home/brendan/...` paths hard-coded inside it.

1. **First boot.** Finish DGX OS setup as user `brendan`, and set the hostname if you like:
   `sudo hostnamectl set-hostname spark-b4e7`. Then apply all updates (DGX Dashboard, or
   `sudo apt update && sudo apt full-upgrade` and `sudo fwupdmgr refresh && sudo fwupdmgr update`), and reboot.
   Check `nvidia-smi` works.

2. **Direct link to Spark 2.** Connect the ConnectX cable to the same port as before, then:
   ```bash
   sudo nmcli connection add type ethernet ifname enp1s0f1np1 con-name connectx-link \
       ipv4.method manual ipv4.addresses 192.168.100.1/30 ipv6.method ignore
   sudo nmcli connection up connectx-link
   ping -c2 192.168.100.2
   ```
   If the interface name differs, `ip -br link` lists the ports. The cable usually shows up on an `enp1s0f*`
   interface.

3. **SSH keys.**
   ```bash
   ssh-keygen -t ed25519 -C "spark1"
   ssh-copy-id brendan@192.168.100.2      # passwordless Spark 1 -> Spark 2 (scripts rely on this)
   cat ~/.ssh/id_ed25519.pub              # add this to GitHub: Settings -> SSH and GPG keys
   ```
   On Spark 2, also run `ssh-copy-id brendan@192.168.100.1` so Spark 2 can reach Spark 1.

4. **Code.** The folder name matters: scripts, docs and Claude's memory all assume `~/rl-gains-in-clinical-tasks`.
   ```bash
   git clone git@github.com:csbrendan/Safe-Reinforcement-Learning-for-Multi-Turn-Clinical-Agents.git ~/rl-gains-in-clinical-tasks
   ```

5. **Data, from Spark 2** (about 5 minutes over the direct link):
   ```bash
   B=192.168.100.2:spark1_backup
   rsync -a --info=progress2 $B/rl-gains-in-clinical-tasks/ ~/rl-gains-in-clinical-tasks/   # checkpoints, results, warranty files
   rsync -a --info=progress2 $B/spark1 $B/v2_bench $B/.secrets ~/
   chmod 700 ~/.secrets && chmod 600 ~/.secrets/*
   mkdir -p ~/.cache/huggingface && rsync -a $B/hf_token ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
   mkdir -p ~/.claude/projects/-home-brendan-rl-gains-in-clinical-tasks/memory
   rsync -a $B/claude_memory/ ~/.claude/projects/-home-brendan-rl-gains-in-clinical-tasks/memory/
   ```
   **From the Mac instead**, if Spark 2 is unavailable: run on the Mac, replacing `spark-b4e7.local` with the new
   unit's address if needed.
   ```bash
   rsync -a --info=progress2 ~/spark1_backup/rl-gains-in-clinical-tasks/ brendan@spark-b4e7.local:rl-gains-in-clinical-tasks/
   rsync -a ~/spark1_backup/v2_bench ~/spark1_backup/.secrets brendan@spark-b4e7.local:
   ```
   Then rebuild the venv (step 6b), because the Mac copy doesn't include it.

6. **Python environment.**
   - **a. Restored copy (normal case).** Check it works:
     ```bash
     ~/spark1/.venv/bin/python -c "import torch, vllm, trl, peft, transformers; print(torch.__version__, torch.cuda.is_available())"
     ```
     It should print `2.11.0+cu128 True`. The venv uses Ubuntu 24.04's system Python 3.12, so it works on a
     standard DGX OS install.
   - **b. Rebuild (if the copy is missing or broken).** Install from the saved list:
     ```bash
     python3 -m venv ~/spark1/.venv
     ~/spark1/.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
         -r ~/venv_freeze_2026-11-14.txt   # copy it from the backup folder first
     ```
     If a package won't install (most likely vllm or flashinfer on Arm), install `torch==2.11.0+cu128` first,
     then `vllm==0.20.1`, then the rest. `~/spark1/README.md` and `~/spark1/docs/` record how it was first set up.
     Key versions: torch 2.11.0+cu128, vllm 0.20.1, trl 1.5.1, peft 0.19.1, transformers 5.8.0,
     liger_kernel 0.8.0, flashinfer 0.6.8.post1.
   - **Don't create a different venv for experiments.** The project rule is to reuse `~/spark1/.venv`.

7. **Models.** The base model the experiments use, `checkpoints/medgemma4b_text`, comes back with step 5. The
   V2 serving copies are already on Spark 2 in `~/v2_bench/models/`. To repopulate the Hugging Face cache, run
   `~/spark1/.venv/bin/hf download google/medgemma-4b-it`. The model is gated, so the token from step 5 must
   belong to an account that has accepted MedGemma's license.

8. **Check everything.**
   ```bash
   cd ~/rl-gains-in-clinical-tasks
   du -sh checkpoints results                       # ~89 GB / ~49 MB
   ls checkpoints | wc -l                           # 33 entries as of 2026-11-14
   ~/spark1/.venv/bin/python -m pytest -q tests     # all tests should pass
   ssh 192.168.100.2 hostname                       # spark-04af
   ```
   **GPU safety:** before any GPU job, read `V2/hw_stability/PROGRESS.md` and the 2-node NCCL notes in
   `framework/procreward/README.md`. Run heavy jobs under `V2/hw_stability/watchdog.py`.

## 3. Refreshing the backup

`rsync` only copies what changed, so re-running is quick. From Spark 1:
```bash
cd ~
rsync -a --info=progress2 rl-gains-in-clinical-tasks/checkpoints rl-gains-in-clinical-tasks/results \
    192.168.100.2:spark1_backup/rl-gains-in-clinical-tasks/
rsync -a rl-gains-in-clinical-tasks/V2/hw_stability/spark1_*.md 192.168.100.2:spark1_backup/rl-gains-in-clinical-tasks/V2/hw_stability/
rsync -a spark1 v2_bench .secrets 192.168.100.2:spark1_backup/
rsync -a .cache/huggingface/token 192.168.100.2:spark1_backup/hf_token
rsync -a .claude/projects/-home-brendan-rl-gains-in-clinical-tasks/memory/ 192.168.100.2:spark1_backup/claude_memory/
~/spark1/.venv/bin/pip freeze | ssh 192.168.100.2 "cat > spark1_backup/venv_freeze_$(date +%F).txt"
```
To refresh the Mac copy, run on the Mac (add `--exclude spark1/.venv` if you pull `spark1/` too):
```bash
rsync -a --info=progress2 brendan@spark-b4e7.local:rl-gains-in-clinical-tasks/checkpoints \
    brendan@spark-b4e7.local:rl-gains-in-clinical-tasks/results ~/spark1_backup/rl-gains-in-clinical-tasks/
```
Rsync never deletes files from the backup unless you add `--delete`. Leave it off, so a mistake on Spark 1 can't
erase the backup.

## 4. Before shipping Spark 1 for an RMA
1. Refresh the backup (section 3) and check the sizes match (section 2, step 8).
2. Keep a second copy off both Sparks, on the Mac or an external drive.
3. Remove credentials from the unit: delete `~/.secrets/`, `~/.cache/huggingface/token` and `~/.ssh/`. Revoke
   that SSH key on GitHub, and remove it from Spark 2's `~/.ssh/authorized_keys`. Rotate the Anthropic API key if
   you want to be thorough.
4. Wipe or reinstall the OS if NVIDIA's instructions say to. NVIDIA is not responsible for data left on a returned
   unit.
