#!/bin/bash
# Description: Downloads and processes a SMALL subset of the ArXiv dataset locally.
# Uses the default limit (currently 50 examples per split) defined in fetch_arxiv.py.

set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting small dataset processing (local output)..."

# Assumes the script is run from the repository root
# Uses default --max_examples_per_split and default output directory ./preprocessed_arxiv/
python fetch_arxiv.py

echo "Small dataset processing script finished." 