#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x ".agent_runtime/venv/bin/python" ]; then
  python3 install.py --torch auto
fi
.agent_runtime/venv/bin/python run_app.py --host 127.0.0.1 --port 7860
