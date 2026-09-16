#!/usr/bin/env bash
# docker build。Synapse の Docker repo は syn77311180。
#   ローカル検証用タグ:  tigersqai26_${TEAM}:${VER}
#   Synapse push 用タグ: docker.synapse.org/syn77311180/tigersqai26_${TEAM}:${VER}
set -euo pipefail
TEAM="${TEAM:-shunsuke}"
VER="${VER:-v4}"
SYN_PROJECT="${SYN_PROJECT:-syn77311180}"
cd "$(dirname "$0")"
LOCAL_TAG="tigersqai26_${TEAM}:${VER}"
SYN_TAG="docker.synapse.org/${SYN_PROJECT}/tigersqai26_${TEAM}:${VER}"
docker build --platform linux/amd64 -t "${LOCAL_TAG}" .
docker tag "${LOCAL_TAG}" "${SYN_TAG}"
echo "built ${LOCAL_TAG}"
echo "tagged ${SYN_TAG}"
echo
echo "push は手動で実行してください:"
echo "  docker login docker.synapse.org      # Synapse のユーザー名 + Personal Access Token"
echo "  docker push ${SYN_TAG}"
