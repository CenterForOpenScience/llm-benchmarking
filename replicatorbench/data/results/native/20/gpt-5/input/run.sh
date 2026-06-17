#!/usr/bin/env bash
set -euo pipefail

# Show PATH for debugging# Show PATH and R locations for debugging
echo "PATH=$PATH"
echo "which R: $(command -v R || echo 'not found')"
echo "which Rscript: $(command -v Rscript || echo 'not found')"
whereis R || true
ls -la /usr/local/bin || true
ls -la /usr/bin || true

# Ensure data directory exists and copy required CSV into /app/data# Ensure data directory exists and copy required CSV into /app/data
mkdir -p /app/data /app/artifacts
SRC="/workspace/replication_data/Kachanoff_Survey_deidentify.csv"
DST="/app/data/Kachanoff_Survey_deidentify.csv"
if [ -f "$SRC" ]; then
  if [ ! -e "$DST" ] || ! [ "$SRC" -ef "$DST" ]; then
    cp -f "$SRC" "$DST"
  else
    echo "Source and destination are the same; skipping copy."
  fi
fi

# Locate an R binary
RBIN=""
if command -v R >/dev/null 2>&1; then
  RBIN="R"
elif command -v Rscript >/dev/null 2>&1; then
  RBIN="Rscript"
elif [ -x "/usr/local/lib/R/bin/R" ]; then
  RBIN="/usr/local/lib/R/bin/R"
elif [ -x "/usr/lib/R/bin/R" ]; then
  RBIN="/usr/lib/R/bin/R"
elif [ -x "/usr/local/bin/R" ]; then
  RBIN="/usr/local/bin/R"
elif [ -x "/usr/bin/R" ]; then
  RBIN="/usr/bin/R"
fi

if [ -z "$RBIN" ]; then
  echo "ERROR: R binary not found in container." >&2
  echo "Contents of /usr/local/lib/R/bin:" >&2
  ls -la /usr/local/lib/R/bin 2>&1 || true
  echo "Contents of /usr/lib/R/bin:" >&2
  ls -la /usr/lib/R/bin 2>&1 || true
  exit 127
fi

echo "Using RBIN=$RBIN"

# Render the RMarkdown
if [ "${RBIN##*/}" = "Rscript" ]; then
  Rscript -e "rmarkdown::render('replication_data/Analysis_updated.Rmd', output_format='html_document', output_dir='/app/artifacts')"
else
  "$RBIN" -q -e "rmarkdown::render('replication_data/Analysis_updated.Rmd', output_format='html_document', output_dir='/app/artifacts')"
fi

echo "Render complete. Artifacts written to /app/artifacts"