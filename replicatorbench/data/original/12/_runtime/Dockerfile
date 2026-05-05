FROM python:3.10
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install pandas>=1.3.0 statsmodels>=0.12.0
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
USER runner
CMD ["bash"]