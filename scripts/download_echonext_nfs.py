"""Download EchoNext full dataset from PhysioNet to NFS Data Drive.

Target: /data/mithunmanivannan/echonext
Source: PhysioNet (https://physionet.org/content/echonext/1.1.1/)
Note: EchoNext is a Restricted Health Data repository requiring a signed DUA.
Pass PhysioNet credentials via command-line or environment variables:
  PHYSIONET_USER and PHYSIONET_PASS
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TARGET_DIR = Path("/data/mithunmanivannan/echonext")
PHYSIONET_URL = "https://physionet.org/files/echonext/1.1.1/"


def main():
    parser = argparse.ArgumentParser(description="Download full EchoNext dataset to NFS")
    parser.add_argument("--user", type=str, default=os.getenv("PHYSIONET_USER"), help="PhysioNet username")
    parser.add_argument("--password", type=str, default=os.getenv("PHYSIONET_PASS"), help="PhysioNet password")
    args = parser.parse_args()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ECHONEXT NFS DOWNLOAD UTILITY")
    print(f"Target Directory: {TARGET_DIR}")
    print(f"Available NFS Space: {os.statvfs(TARGET_DIR).f_bavail * os.statvfs(TARGET_DIR).f_frsize / (1024**3):.1f} GB")
    print(f"Source URL: {PHYSIONET_URL}")
    print("=" * 80)

    if not args.user or not args.password:
        print("\n[!] PhysioNet credentials required for Restricted-Access EchoNext.")
        print("Please provide credentials via:")
        print("  python scripts/download_echonext_nfs.py --user <username> --password <password>")
        print("Or set environment variables PHYSIONET_USER and PHYSIONET_PASS.")
        return

    cmd = [
        "wget",
        "-r",
        "-N",
        "-c",
        "-np",
        f"--user={args.user}",
        f"--password={args.password}",
        "-P",
        str(TARGET_DIR),
        PHYSIONET_URL,
    ]
    print(f"Executing: wget -r -N -c -np --user={args.user} -P {TARGET_DIR} {PHYSIONET_URL}")
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print(f"\n[+] Successfully downloaded EchoNext to {TARGET_DIR}!")
    else:
        print(f"\n[-] Download exited with status code {res.returncode}")


if __name__ == "__main__":
    main()
