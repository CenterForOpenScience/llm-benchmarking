FROM python:3.10
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git wget ca-certificates && rm -rf /var/lib/apt/lists/*
RUN command -v python3 || (apt-get update && apt-get install -y python3 python3-pip)
RUN pip3 install --no-cache-dir pandas>=1.4.0 statsmodels>=0.13.0 seaborn>=0.11.0 matplotlib>=3.4.0 scikit-learn
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
CMD ["bash"]