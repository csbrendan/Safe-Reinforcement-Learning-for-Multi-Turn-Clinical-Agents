#!/usr/bin/env bash
# V2 GPU VM lifecycle, run from Spark 2 (needs the Azure CLI signed in). See V2/azure/README.md.
#
#   APPROVED=yes bash V2/azure/vm.sh create    # starts billing (~$6.98/h for 1x H100 NVL on-demand)
#   bash V2/azure/vm.sh push                   # copy repo-external pieces (SD-Bench clone, secrets) to the VM
#   bash V2/azure/vm.sh setup                  # run setup_remote.sh on the VM (drivers check, envs, model)
#   bash V2/azure/vm.sh ssh | status | ip
#   APPROVED=yes bash V2/azure/vm.sh delete    # stops billing for compute AND disk; the GPU is released
#
# Settings (environment): LOC (westus3), RG (v2-gpu-rg), NAME (v2-h100-1), SIZE (Standard_NC40ads_H100_v5),
# PRIORITY (Regular | Spot; Spot only for restartable evaluation work), DISK_GB (1024), SSH_KEY (~/.ssh/id_ed25519.pub),
# SSH_SOURCE (your public IP/CIDR for the SSH rule; auto-detected if unset).
#
# Holding policy (V2/skyrl_plan.md): create only when the run window opens, keep the VM allocated for the window
# (don't "Stop (deallocate)" mid-window: restarting can fail if the region is out of GPUs), delete at the end.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

LOC="${LOC:-westus3}"
RG="${RG:-v2-gpu-rg}"
NAME="${NAME:-v2-h100-1}"
SIZE="${SIZE:-Standard_NC40ads_H100_v5}"
PRIORITY="${PRIORITY:-Regular}"
DISK_GB="${DISK_GB:-1024}"
IMAGE="${IMAGE:-microsoft-dsvm:ubuntu-hpc:2404:latest}"   # Azure HPC image: NVIDIA driver + CUDA preinstalled
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519.pub}"
USER_VM="azureuser"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

need_approval() {
  if [[ "${APPROVED:-}" != "yes" ]]; then
    echo "Refusing: '$1' changes billing. Re-run with APPROVED=yes once Brendan has approved it." >&2
    exit 2
  fi
}

ip() { az vm show -d -g "$RG" -n "$NAME" --query publicIps -o tsv; }

case "${1:-}" in
  create)
    need_approval create
    src="${SSH_SOURCE:-$(curl -fsS https://api.ipify.org)/32}"
    az group create -n "$RG" -l "$LOC" --tags purpose=v2-sdbench -o none
    spot=()
    if [[ "$PRIORITY" == "Spot" ]]; then spot=(--eviction-policy Deallocate --max-price -1); fi
    az vm create -g "$RG" -n "$NAME" -l "$LOC" --size "$SIZE" --image "$IMAGE" --priority "$PRIORITY" "${spot[@]}" \
      --security-type Standard --os-disk-size-gb "$DISK_GB" --storage-sku Premium_LRS \
      --admin-username "$USER_VM" --ssh-key-values "$SSH_KEY" --authentication-type ssh \
      --public-ip-sku Standard --nsg-rule NONE --tags purpose=v2-sdbench -o table
    az network nsg rule create -g "$RG" --nsg-name "${NAME}NSG" -n ssh-from-spark2 --priority 100 \
      --source-address-prefixes "$src" --destination-port-ranges 22 --protocol Tcp --access Allow -o none
    echo "VM up at $(ip). SSH allowed only from $src. Billing has started."
    ;;
  push)
    h="$USER_VM@$(ip)"
    ssh -o StrictHostKeyChecking=accept-new "$h" 'mkdir -p ~/v2_bench ~/.secrets && chmod 700 ~/.secrets'
    # SD-Bench clone (incl. the public chargemaster xlsx). Never from ~/spark1_backup.
    rsync -a --info=stats1 "$HOME/v2_bench/sdbench/" "$h:v2_bench/sdbench/"
    # This repo at the current commit (the VM clones it too; this carries uncommitted configs if any).
    rsync -a --info=stats1 --exclude checkpoints --exclude results --exclude '.git' "$REPO/" "$h:rl-gains-in-clinical-tasks/"
    # Secrets: copied, never printed. Hugging Face token for the gated MedGemma weights.
    scp -q "$HOME/.secrets/azure.env" "$h:.secrets/azure.env"
    ssh "$h" 'chmod 600 ~/.secrets/azure.env && mkdir -p ~/.cache/huggingface'
    scp -q "$HOME/.cache/huggingface/token" "$h:.cache/huggingface/token"
    ssh "$h" 'chmod 600 ~/.cache/huggingface/token'
    echo "pushed to $h"
    ;;
  setup)
    ssh -t "$USER_VM@$(ip)" 'bash ~/rl-gains-in-clinical-tasks/V2/azure/setup_remote.sh'
    ;;
  ssh)    ssh "$USER_VM@$(ip)" ;;
  ip)     ip ;;
  status) az vm show -d -g "$RG" -n "$NAME" --query "{power:powerState,size:hardwareProfile.vmSize,ip:publicIps,loc:location}" -o table ;;
  delete)
    need_approval delete
    az group delete -n "$RG" --yes
    echo "deleted resource group $RG (VM, disk, IP, NSG). The GPU is released."
    ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
