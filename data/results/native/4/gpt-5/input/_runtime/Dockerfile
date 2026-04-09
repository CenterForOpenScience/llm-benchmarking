FROM rocker/tidyverse:4.3.1
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git wget ca-certificates && rm -rf /var/lib/apt/lists/*
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git r-base-core && rm -rf /var/lib/apt/lists/*
RUN command -v R || (apt-get update && apt-get install -y r-base)
RUN R -q -e 'install.packages(c("readr","dplyr","psych"), repos="https://cloud.r-project.org")'
WORKDIR /workspace
RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app
# Ensure Rscript is available on PATH expected by orchestrator before switching user# Ensure Rscript is available on PATH expected by orchestrator before switching user
ENV PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
RUN if [ -x /usr/local/bin/Rscript ]; then cp -f /usr/local/bin/Rscript /usr/bin/Rscript || true; fi
USER runnerUSER runnerUSER runnerUSER runnerUSER runner
CMD ["bash"]