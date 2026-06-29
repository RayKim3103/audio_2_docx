#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "This will delete packages, models, and caches under .agent_runtime"
read -rp "Continue? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
  rm -rf .agent_runtime
fi
