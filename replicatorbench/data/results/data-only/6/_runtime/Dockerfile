FROM python:3.10-slim
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git wget ca-certificates && rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y bash locales build-essential gcc g++ make zlib1g-dev libbz2-dev && rm -rf /var/lib/apt/lists/*
RUN command -v python3 || (apt-get update && apt-get install -y python3 python3-pip)
RUN pip3 install --no-cache-dir pip>=23.0.0 setuptools>=68.0.0 wheel>=0.40.0 numpy>=1.23.0 pandas>=1.5.0 statsmodels>=0.13.0 pyreadstat==1.2.7
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
CMD ["bash"]