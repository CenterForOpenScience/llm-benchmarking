FROM python:3.10-slim

# Install essential system packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git wget ca-certificates make build-essential \
    && rm -rf /var/lib/apt/lists/*

# Optionally install R if required by the original code
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    r-base \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 and pip are available
RUN command -v python3 || (apt-get update && apt-get install -y python3 python3-pip)

# Python dependencies (exclude rpy2 to avoid building R bindings in this image)
RUN pip3 install --no-cache-dir pandas numpy scipy statsmodels pyreadr scikit-learn matplotlib seaborn

WORKDIR /workspace

# Create non-root user and app directories
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
# Default command: run the Python replication entrypoint
CMD ["python3", "/workspace/replication_data/run_replication.py"]