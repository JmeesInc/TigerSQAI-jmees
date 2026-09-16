#!/usr/bin/env bash
# 実験コード (py/sh/yaml のみ、results/ 除外) を tgz に固める → tiger_code.tgz
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${DIR}/../.." && pwd)"
OUT="${DIR}/tiger_code.tgz"
cd "${REPO}"
find workspace -maxdepth 2 -type f \( -name "*.py" -o -name "*.sh" -o -name "*.yaml" \) \
  | grep -vE "results|vast_ops" \
  | tar -czf "${OUT}" -T -
ls -la --block-size=K "${OUT}"
