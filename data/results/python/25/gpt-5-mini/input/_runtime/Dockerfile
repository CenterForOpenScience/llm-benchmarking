FROM rocker/verse

# Install basics and python packages system-wide to avoid PEP 668 issues
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git wget ca-certificates python3 python3-pip python3-pandas python3-numpy python3-openpyxl \
    && rm -rf /var/lib/apt/lists/*

# Also ensure pip is available and can install packages in user site if needed
RUN python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
CMD ["bash"]