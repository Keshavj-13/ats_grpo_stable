#!/usr/bin/env bash
# Mirror raw GitHub sources from four OpenEnv / PyTorch hackathon–adjacent repos
# into reference/openenv_hackathon/ for offline diffing. Run from repo root:
#   bash scripts/fetch_openenv_hackathon_refs.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/reference/openenv_hackathon"
mkdir -p "$OUT"/{kube_sre_gym,openenv_hack_yash,openoffice_rl,meta_openenv_hack}

fetch() {
  local url="$1" dest="$2"
  echo "  -> $dest"
  curl -fsSL "$url" -o "$dest"
}

echo "Writing to $OUT"

# --- 1) sid-rp/kube-sre-gym (1st-place stack: GRPO + OpenEnv + vLLM patterns) ---
fetch "https://raw.githubusercontent.com/sid-rp/kube-sre-gym/main/train.py" \
  "$OUT/kube_sre_gym/train.py"
fetch "https://raw.githubusercontent.com/sid-rp/kube-sre-gym/main/openenv.yaml" \
  "$OUT/kube_sre_gym/openenv.yaml"
fetch "https://raw.githubusercontent.com/sid-rp/kube-sre-gym/main/client.py" \
  "$OUT/kube_sre_gym/client.py"

# --- 2) YashJoshi2109/OpenEnv_Hack (reward outside env, GRPO colab, negotiation) ---
fetch "https://raw.githubusercontent.com/YashJoshi2109/OpenEnv_Hack/main/reward.py" \
  "$OUT/openenv_hack_yash/reward.py"
fetch "https://raw.githubusercontent.com/YashJoshi2109/OpenEnv_Hack/main/train_colab.py" \
  "$OUT/openenv_hack_yash/train_colab.py"
fetch "https://raw.githubusercontent.com/YashJoshi2109/OpenEnv_Hack/main/openenv.yaml" \
  "$OUT/openenv_hack_yash/openenv.yaml"
fetch "https://raw.githubusercontent.com/YashJoshi2109/OpenEnv_Hack/main/server/negotiation_environment.py" \
  "$OUT/openenv_hack_yash/negotiation_environment.py"

# --- 3) bvsbharat/OpenOfficeRL (multi-agent train loop; real code under office_os/) ---
fetch "https://raw.githubusercontent.com/bvsbharat/OpenOfficeRL/main/main.py" \
  "$OUT/openoffice_rl/main_root.py"
fetch "https://raw.githubusercontent.com/bvsbharat/OpenOfficeRL/main/office_os/train_loop.py" \
  "$OUT/openoffice_rl/office_os_train_loop.py"
fetch "https://raw.githubusercontent.com/bvsbharat/OpenOfficeRL/main/office_os/openenv.yaml" \
  "$OUT/openoffice_rl/office_os_openenv.yaml"

# --- 4) harishchaurasia/Meta_OpenEnv_PyTorch_Hack (masked discrete actions + GRPO skeleton) ---
fetch "https://raw.githubusercontent.com/harishchaurasia/Meta_OpenEnv_PyTorch_Hack/main/train_skeleton.py" \
  "$OUT/meta_openenv_hack/train_skeleton.py"
fetch "https://raw.githubusercontent.com/harishchaurasia/Meta_OpenEnv_PyTorch_Hack/main/openenv.yaml" \
  "$OUT/meta_openenv_hack/openenv.yaml"

cat > "$OUT/SOURCES.txt" <<'EOF'
Raw mirrors (main branch). Re-fetch after upstream changes:

https://github.com/sid-rp/kube-sre-gym
https://github.com/YashJoshi2109/OpenEnv_Hack
https://github.com/bvsbharat/OpenOfficeRL
https://github.com/harishchaurasia/Meta_OpenEnv_PyTorch_Hack

What we already adopted in-tree (see training/train_grpo.py top):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  TRL_EXPERIMENTAL_SILENCE=1

Ideas worth manual porting (not auto-merged — domain mismatch):
  - kube: loss_type=dapo, beta, mask_truncated_completions in GRPOConfig (verify Unsloth TRL build first)
  - Yash: reward module kept strictly outside env (we use training/reward_functions.py — same idea)
  - harish: action masking / discrete scoring (ATC needs rich JSON; optional for auxiliary "format" head only)
  - OpenOfficeRL: data generation vs train_loop split (compare to training/dataset.py + train_grpo.py)
EOF

echo "Done. Index: $OUT/SOURCES.txt"
