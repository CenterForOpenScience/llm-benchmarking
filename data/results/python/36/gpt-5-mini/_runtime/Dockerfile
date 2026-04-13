FROM python:3.10-slim
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git wget ca-certificates build-essential graphviz libgraphviz-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
# Install python packages
RUN pip install --no-cache-dir pandas>=1.4 numpy>=1.22 semopy>=2.0.0 scipy>=1.7 statsmodels>=0.13 matplotlib>=3.5 networkx>=2.8 graphviz>=0.20
CMD ["bash"]