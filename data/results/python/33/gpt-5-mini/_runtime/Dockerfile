FROM python:3.9-slimFROM python:3.9-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt /app/data/requirements.txt
RUN pip install --no-cache-dir -r /app/data/requirements.txt
CMD ["bash"]CMD ["bash"]CMD ["bash"]