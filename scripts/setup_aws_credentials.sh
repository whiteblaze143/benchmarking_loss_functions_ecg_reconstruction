#!/usr/bin/env bash
# Script to verify and configure AWS credentials and check S3 permissions for BDSP S3 access points.
set -euo pipefail

AWS_ACCOUNT="711012205142"
ACCESS_POINT_AC="s3://bdsp-credentialed-ac-psbrsg8wcmky4w5tbtn3b31yh4otause1b-s3alias"
ACCESS_POINT_PR="s3://bdsp-credentialed-pr-pakus5b5ruieu8mruai3jgxtcu56suse1b-s3alias"

echo "=== 1. Checking AWS STS Caller Identity ==="
if ! aws sts get-caller-identity --output json; then
  echo "[ERROR] Failed to obtain caller identity."
  exit 1
fi

echo ""
echo "=== 2. Probing BDSP S3 Access Points ==="
echo "Testing: ${ACCESS_POINT_AC}/"
if aws s3 ls "${ACCESS_POINT_AC}/" --region us-east-1; then
  echo "[SUCCESS] Successfully listed Credentialed Access Point!"
else
  echo "[WARNING] Access point returned an error. Note: In AWS IAM, user 'mithunm' needs an identity policy (e.g. AmazonS3ReadOnlyAccess) attached in the AWS Console."
fi

echo ""
echo "Testing: ${ACCESS_POINT_PR}/"
if aws s3 ls "${ACCESS_POINT_PR}/" --region us-east-1; then
  echo "[SUCCESS] Successfully listed Projects Access Point!"
else
  echo "[WARNING] Projects access point returned an error."
fi
