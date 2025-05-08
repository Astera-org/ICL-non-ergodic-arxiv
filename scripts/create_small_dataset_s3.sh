#!/bin/bash
# Description: Ensures the SMALL subset of the ArXiv dataset exists on S3.
# 1. Attempts to download from S3 (using the specified S3_PREFIX).
# 2. If download fails, processes the SMALL dataset locally and uploads it.

# Source .env file if it exists
if [ -f ".env" ]; then export $(grep -v '^#' .env | xargs); fi

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration --- 
# !!! MUST SET THESE VARIABLES BEFORE RUNNING !!!
S3_BUCKET="obelisk-simplex"  # Replace with your target S3 bucket name
S3_PREFIX="non-ergodic-arxiv/preprocessed_arxiv_small" # Optional: Prefix within bucket (project folder/small dataset)
# Ensure your AWS credentials are configured (e.g., via ~/.aws/credentials, env vars, or IAM role)
# --------------------

echo "Ensuring small dataset exists at s3://${S3_BUCKET}/${S3_PREFIX}..."

if [ "$S3_BUCKET" == "your-s3-bucket-name-here" ]; then
  echo "Error: Please set the S3_BUCKET variable in this script before running."
  exit 1
fi

echo "Shell script will now call fetch_arxiv.py."
echo "fetch_arxiv.py is configured to perform the following steps if needed:"
echo "1. Attempt to download preprocessed data from s3://${S3_BUCKET}/${S3_PREFIX}"
echo "2. If download fails or data is incomplete, it will process the SMALL dataset locally."
echo "3. If data was processed locally, it will attempt to upload it to s3://${S3_BUCKET}/${S3_PREFIX}"
echo "--- Output from fetch_arxiv.py starts below ---"

# Assumes the script is run from the repository root
# Attempts download first. If successful, local processing and upload are skipped.
# If download fails, it processes locally (using default max_examples) and then uploads.
python fetch_arxiv.py \
  --download_from_s3 \
  --upload_to_s3 \
  --s3_bucket "$S3_BUCKET" \
  --s3_prefix "$S3_PREFIX"
  # Note: We don't need --max_examples_per_split here, as the default in fetch_arxiv.py handles the small dataset size.
  # If the download fails, the script proceeds with the default local processing limit.
echo "--- Output from fetch_arxiv.py finished ---"
echo "fetch_arxiv.py execution completed by the shell script."

echo "Ensure small dataset script finished." 