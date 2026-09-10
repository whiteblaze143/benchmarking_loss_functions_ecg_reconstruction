#!/usr/bin/env python3
"""
Test and explore BDSP (Brain Data Science Platform) S3 access points for HEEDB.
Verifies AWS credentials, tests Signature Version 4 against the authorized access points,
and lists the root structure of the Harvard-Emory ECG Database (HEEDB).
"""

import os
import subprocess
import sys

ACCESS_POINT_AC = "s3://bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias"
ACCESS_POINT_PR = "s3://bdsp-credentialed-pr-pakus5b5ruieu8mruai3jgxtcu56suse1b-s3alias"
EXPECTED_ACCOUNT = "711012205142"


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    return proc.returncode, out


def check_identity():
    print("=== Checking AWS Identity ===")
    code, out = run_cmd(["aws", "sts", "get-caller-identity", "--output", "json"])
    if code != 0:
        print("ERROR: AWS credentials not configured or invalid.")
        print(f"Details:\n{out}")
        return False

    print(f"Active Identity:\n{out}")
    if EXPECTED_ACCOUNT in out:
        print(f"SUCCESS: Authenticated with authorized Account ID {EXPECTED_ACCOUNT}!")
    else:
        print(f"WARNING: Current account differs from authorized Account ID {EXPECTED_ACCOUNT}.")
    return True


def probe_access_points():
    print("\n=== Probing BDSP S3 Access Points ===")
    for name, ap in [("Credentialed Access Point", ACCESS_POINT_AC), ("Projects Access Point", ACCESS_POINT_PR)]:
        print(f"\nProbing {name}: {ap}/")
        code, out = run_cmd(["aws", "s3", "ls", f"{ap}/", "--region", "us-east-1"])
        if code == 0:
            print(f"SUCCESS! Listing for {ap}/:\n{out}")
        else:
            print(f"FAILED to list {ap}/. Error:\n{out}")


if __name__ == "__main__":
    has_creds = check_identity()
    if has_creds:
        probe_access_points()
    else:
        print("\nPlease configure AWS credentials using 'aws configure' or set:")
        print("  export AWS_ACCESS_KEY_ID='...'")
        print("  export AWS_SECRET_ACCESS_KEY='...'")
        print("  export AWS_DEFAULT_REGION='us-east-1'")
        sys.exit(1)
