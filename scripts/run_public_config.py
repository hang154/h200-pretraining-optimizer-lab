#!/usr/bin/env python3
"""Translate an audited JSON config into the public-corpus trainer CLI."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--out", required=True)
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
command = [sys.executable, "scripts/train_public_corpus.py", "--out", args.out]
for key, value in config.items():
    if key == "tag" or value is not None:
        command.extend(["--" + key.replace("_", "-"), str(value)])
raise SystemExit(subprocess.call(command))
