from __future__ import annotations

import argparse
import json
from pathlib import Path

from repecg.common import load_config
from repecg.common.ptbxl import PTBXLStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = PTBXLStore(load_config(args.config)).cohort_manifest()
    rendered = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
