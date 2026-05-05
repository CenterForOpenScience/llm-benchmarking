#!/usr/bin/env bash
set -euo pipefail
echo "Starting replication run via bash wrapper."
echo "PATH is: $PATH"
echo "R version:" && R --version || true
R -q -f "/workspace/replication_data/Fitzgerald 2018 Script_clean v2.R"
echo "Replication script completed."