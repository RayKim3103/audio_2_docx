from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["AUDIO_DOCX_PROJECT_ROOT"] = str(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from meeting_docx_agent.paths import OUTPUT_DIR, configure_environment

configure_environment()


def parse_args():
    p = argparse.ArgumentParser(description="Local Meeting Audio to DOCX Agent")
    p.add_argument("--host", default="127.0.0.1", help="Bind host. Use 0.0.0.0 for LAN.")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--lan", action="store_true", help="Shortcut for --host 0.0.0.0")
    p.add_argument("--share", action="store_true", help="Create a public Gradio share link. Not recommended for secure local use.")
    p.add_argument("--auth", default="", help="Basic auth in user:password format. Recommended for LAN mode.")
    p.add_argument("--inbrowser", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    from meeting_docx_agent.ui import build_ui
    auth = None
    if args.auth and ":" in args.auth:
        user, pw = args.auth.split(":", 1)
        auth = (user, pw)
    server_name = "0.0.0.0" if args.lan else args.host
    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=10).launch(
        server_name=server_name,
        server_port=args.port,
        share=args.share,
        auth=auth,
        inbrowser=args.inbrowser,
        show_error=True,
        allowed_paths=[str(OUTPUT_DIR.resolve())],
        enable_monitoring=False,
    )


if __name__ == "__main__":
    main()
