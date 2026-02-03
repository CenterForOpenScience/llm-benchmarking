FROM python:3.10-slim
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git wget ca-certificates && rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y wget git && rm -rf /var/lib/apt/lists/*
RUN command -v python3 || (apt-get update && apt-get install -y python3 python3-pip)
RUN pip3 install --no-cache-dir pandas==2.1.4 numpy==1.26.3 statsmodels==0.14.1 pyreadstat==1.2.6
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
CMD ["bash"]