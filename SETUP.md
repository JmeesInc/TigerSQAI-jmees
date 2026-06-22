# Setup on another machine

This repo contains **code only**. Data, weights and vendored external repos are NOT in git.

## 1. Clone
```bash
git clone <repo-url> TigerSQAI && cd TigerSQAI
```

## 2. Data
`data/` symlinks to `/mnt/data/data4/shared/miccai/EndoVis2026/tiger`. Mount and `ln -s`, or rsync from source:
```bash
rsync -avP <user>@dl1:/mnt/data/data4/shared/miccai/EndoVis2026/tiger/ ./data_local/ && ln -s "$PWD/data_local" data
```

## 3. Weights / vendored references
`reference/`, vendored nested repos, and `*.pth/*.pt/*.safetensors/*.ckpt` are git-ignored.
Pull what you need directly from the source machine (server-to-server, NOT via laptop):
```bash
rsync -avP <user>@dl1:/data4/src/shunsuke/MICCAI2026/TigerSQAI/reference/ ./reference/
```

## 4. Python env
No dependency manifest is committed yet. Set up the env on the target machine (e.g. `uv venv` + install), and consider committing a `requirements.txt` once stable.

## Notes
- Experiment **code** under `workspace/<exp>/` IS tracked (`.py/.sh/.yaml/.md`).
- Artifacts (checkpoints, big JSON dumps, downloads, outputs, logs) are git-ignored — regenerate or rsync.
- Reproducibility metadata excluded by the whitelist (e.g. fold splits) can be regenerated, or pulled via `scp dl1:/data4/src/shunsuke/MICCAI2026/TigerSQAI/...`.
