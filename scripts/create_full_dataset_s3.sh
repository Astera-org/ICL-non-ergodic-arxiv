#!/bin/bash
# Description: Ensures the FULL ArXiv dataset exists on S3.
# 1. Attempts to download from S3.
# 2. If download fails, processes the FULL dataset locally and uploads it.
# Warning: Local processing takes significant time and disk space.

# Source .env file if it exists
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration --- 
# !!! MUST SET THESE VARIABLES BEFORE RUNNING !!!
S3_BUCKET="obelisk-simplex"  # Replace with your target S3 bucket name
S3_PREFIX="non-ergodic-arxiv/preprocessed_arxiv"      # Optional: Prefix (folder) within the bucket (defaults to 'preprocessed_arxiv')
# Ensure your AWS credentials are configured (e.g., via ~/.aws/credentials, env vars, or IAM role)
# --------------------

echo "Ensuring full dataset exists at s3://${S3_BUCKET}/${S3_PREFIX}..."

if [ "$S3_BUCKET" == "your-s3-bucket-name-here" ]; then
  echo "Error: Please set the S3_BUCKET variable in this script before running."
  exit 1
fi

# Assumes the script is run from the repository root
# Attempts download first. If successful, local processing and upload are skipped.
# If download fails, it processes locally (--full_dataset) and then uploads (--upload_to_s3).
python fetch_arxiv.py \
  --download_from_s3 \
  --full_dataset \
  --upload_to_s3 \
  --s3_bucket "$S3_BUCKET" \
  --s3_prefix "$S3_PREFIX"

echo "Ensure full dataset script finished." 