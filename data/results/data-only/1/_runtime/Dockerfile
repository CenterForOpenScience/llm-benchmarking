FROM python:3.10-slim
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git wget ca-certificates && rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y bash && rm -rf /var/lib/apt/lists/*
RUN command -v python3 || (apt-get update && apt-get install -y python3 python3-pip)
RUN python -m pip install --no-cache-dir "numpy>=1.24.0,<2.0.0" "pandas>=1.5.0,<2.2.0" "statsmodels>=0.13.5,<0.15.0" "patsy>=0.5.3,<0.6.0"
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
CMD ["bash"]